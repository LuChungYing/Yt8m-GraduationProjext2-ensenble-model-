#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Binary for evaluating Tensorflow models on the YouTube-8M dataset."""

import time
import tempfile
import glob
import heapq
import json
import numpy
import eval_util
import losses
import ensemble_level_models
import readers
import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import utils
from six.moves import urllib

import numpy as np

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("model_checkpoint_path", None,
                      "The file to load the model files from. ")
  flags.DEFINE_string("output_file", "",
                      "The file to save the predictions to.")
  flags.DEFINE_string(
      "input_data_patterns", "",
      "File globs defining the evaluation dataset in tensorflow.SequenceExample format.")
  flags.DEFINE_string(
      "input_data_pattern", None,
      "File globs for original model input.")
  flags.DEFINE_string("feature_names", "predictions", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "3862", "Length of the feature vectors.")

  # Model flags.
  flags.DEFINE_string(
      "model", "LogisticModel",
      "Which architecture to use for the model.")
  flags.DEFINE_integer("batch_size", 256,
                       "How many examples to process per batch.")

  # Other flags.
  flags.DEFINE_boolean("run_once", True, "Whether to run eval only once.")
  flags.DEFINE_integer("top_k", 20, "How many predictions to output per video.")
  
  flags.DEFINE_bool(
      "segment_labels", False,
      "If set, then --input_data_pattern must be frame-level features (but with"
      " segment_labels). Otherwise, --input_data_pattern must be aggregated "
      "video-level features. The model must also be set appropriately (i.e. to "
      "read 3D batches VS 4D batches.")
  flags.DEFINE_integer("segment_max_pred", 100000,
                       "Limit total number of segment outputs per entity.")
  flags.DEFINE_string(
      "segment_label_ids_file",
      "https://raw.githubusercontent.com/google/youtube-8m/master/segment_label_ids.csv",
      "The file that contains the segment label ids.")
  flags.DEFINE_string(
      "input_model_tgz", "",
      "If given, must be path to a .tgz file that was written "
      "by this binary using flag --output_model_tgz. In this "
      "case, the .tgz file will be untarred to "
      "--untar_model_dir and the model will be used for "
      "inference.")
  

def format_lines(video_ids, predictions, top_k):
  batch_size = len(video_ids)
  for video_index in range(batch_size):
    top_indices = numpy.argpartition(predictions[video_index], -top_k)[-top_k:]
    line = [(class_index, predictions[video_index][class_index])
            for class_index in top_indices]
    line = sorted(line, key=lambda p: -p[1])
    yield video_ids[video_index].decode('utf-8') + "," + " ".join("%i %f" % pair
                                                  for pair in line) + "\n"


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)


def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=256):
  logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
  with tf.name_scope("eval_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find the evaluation files.")
    logging.info("number of evaluation files: " + str(len(files)))
    files.sort()
    filename_queue = tf.train.string_input_producer(
        files, shuffle=False, num_epochs=1)
    eval_data = reader.prepare_reader(filename_queue)
    return tf.train.batch(
        eval_data,
        batch_size=batch_size,
        capacity=3 * batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def build_graph(all_readers,
                all_data_patterns,
                input_reader,
                input_data_pattern,
                model,
                batch_size=256):
  """Creates the Tensorflow graph for evaluation.

  Args:
    all_readers: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    all_data_patterns: glob path to the evaluation data files.
    batch_size: How many examples to process at a time.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")

  model_input_raw_tensors = []
  labels_batch_tensor = None
  video_id_batch = None
  for reader, data_pattern in zip(all_readers, all_data_patterns):
    unused_video_id, model_input_raw, labels_batch, unused_num_frames = (
        get_input_data_tensors(
            reader,
            data_pattern,
            batch_size=batch_size))
    if labels_batch_tensor is None:
      labels_batch_tensor = labels_batch
    if video_id_batch is None:
      video_id_batch = unused_video_id
    model_input_raw_tensors.append(tf.expand_dims(model_input_raw, axis=2))

  original_input = None
  if input_data_pattern is not None:
    unused_video_id, original_input, unused_labels_batch, unused_num_frames = (
        get_input_data_tensors(
            input_reader,
            input_data_pattern,
            batch_size=batch_size))

  model_input = tf.concat(model_input_raw_tensors, axis=2)
  labels_batch = labels_batch_tensor

  with tf.name_scope("model"):
    result = model.create_model(model_input,
                                labels=labels_batch,
                                vocab_size=reader.num_classes,
                                original_input=original_input,
                                is_training=False)
    predictions = result["predictions"]

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("predictions", predictions)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("video_id_batch", video_id_batch)
  tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))


def inference_loop(video_id_batch, prediction_batch, label_batch,
              saver, out_file_location):

  top_k = FLAGS.top_k
  with tf.Session() as sess, gfile.Open(out_file_location, "w+") as out_file:
    checkpoint = FLAGS.model_checkpoint_path
    if checkpoint:
      logging.info("Loading checkpoint for eval: " + checkpoint)
      saver.restore(sess, checkpoint)
      global_step_val = checkpoint.split("/")[-1].split("-")[-1]
    else:
      logging.info("No checkpoint file found.")
      return global_step_val

    sess.run([tf.local_variables_initializer()])

    # Start the queue runners.
    fetches = [video_id_batch, prediction_batch]
    coord = tf.train.Coordinator()
    if FLAGS.segment_labels:
      final_out_file = out_file
      out_file = tempfile.NamedTemporaryFile()
      logging.info(
          "Segment temp prediction output will be written to temp file: %s",
          out_file.name)
      
      if FLAGS.segment_label_ids_file:
        whitelisted_cls_mask = np.zeros((prediction_batch.get_shape()[-1],),
                                        dtype=np.float32)
        segment_label_ids_file = FLAGS.segment_label_ids_file
        if segment_label_ids_file.startswith("http"):
          logging.info("Retrieving segment ID whitelist files from %s...",
                       segment_label_ids_file)
          segment_label_ids_file, _ = urllib.request.urlretrieve(
              segment_label_ids_file)
        with tf.io.gfile.GFile(segment_label_ids_file) as fobj:
          for line in fobj:
            try:
              cls_id = int(line)
              whitelisted_cls_mask[cls_id] = 1.
            except ValueError:
              # Simply skip the non-integer line.
              continue
      
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True,
            start=True))
      logging.info("enter eval_once loop global_step_val = %s. ",
                   global_step_val)

      num_examples_processed = 0
      start_time = time.time()
      out_file.write("VideoId,LabelConfidencePairs\n")

      while not coord.should_stop():
        batch_start_time = time.time()

        video_id_val, predictions_val = sess.run(fetches)

        now = time.time()
        num_examples_processed += len(video_id_val)
        logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))
        for line in format_lines(video_id_val, predictions_val, top_k):
          out_file.write(line)
        out_file.flush()

    except tf.errors.OutOfRangeError as e:
      logging.info('Done with inference. The output file was written to ' + out_file_location)
    except Exception as e:  # pylint: disable=broad-except
      logging.info("Unexpected exception: " + str(e))
      coord.request_stop(e)
    ############################
    if FLAGS.segment_labels:
        # Re-read the file and do heap sort.
        # Create multiple heaps.
        logging.info("Post-processing segment predictions...")
        heaps = {}
        out_file.seek(0, 0)
        for line in out_file:
          segment_id, preds = line.decode("utf8").split(",")
          if segment_id == "VideoId":
            # Skip the headline.
            continue
          preds = preds.split(" ")
          pred_cls_ids = [int(preds[idx]) for idx in range(0, len(preds), 2)]
          pred_cls_scores = [
              float(preds[idx]) for idx in range(1, len(preds), 2)
          ]
          for cls, score in zip(pred_cls_ids, pred_cls_scores):
            if not whitelisted_cls_mask[cls]:
              # Skip non-whitelisted classes.
              continue
            if cls not in heaps:
              heaps[cls] = []
            if len(heaps[cls]) >= FLAGS.segment_max_pred:
              heapq.heappushpop(heaps[cls], (score, segment_id))
            else:
              heapq.heappush(heaps[cls], (score, segment_id))
        logging.info("Writing sorted segment predictions to: %s",
                     final_out_file.name)
        final_out_file.write("Class,Segments\n")
        for cls, cls_heap in heaps.items():
          cls_heap.sort(key=lambda x: x[0], reverse=True)
          final_out_file.write("%d,%s\n" %
                               (cls, " ".join([x[1] for x in cls_heap])))
        final_out_file.close()
        out_file.close()
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def inference():
  tf.set_random_seed(0)  # for reproducibility
  with tf.Graph().as_default():
    # convert feature_names and feature_sizes to lists of values
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        FLAGS.feature_names, FLAGS.feature_sizes)

    # prepare a reader for each single model prediction result
    all_readers = []

    all_patterns = FLAGS.input_data_patterns
    all_patterns = map(lambda x: x.strip(), all_patterns.strip().strip(",").split(","))
    for i in xrange(len(all_patterns)):
      reader = readers.EnsembleReader(
          feature_names=feature_names, feature_sizes=feature_sizes)
      all_readers.append(reader)

    input_reader = None
    input_data_pattern = None
    if FLAGS.input_data_pattern is not None:
      input_reader = readers.EnsembleReader(
          feature_names=["input"], feature_sizes=[1024+128])
      input_data_pattern = FLAGS.input_data_pattern

    model = find_class_by_name(FLAGS.model, [ensemble_level_models])()

    if FLAGS.input_data_patterns is "":
      raise IOError("'input_data_patterns' was not specified. " +
                     "Nothing to evaluate.")

    build_graph(
        all_readers=all_readers,
        all_data_patterns=all_patterns,
        input_reader=input_reader,
        input_data_pattern=input_data_pattern,
        model=model,
        batch_size=FLAGS.batch_size)

    logging.info("built evaluation graph")
    video_id_batch = tf.get_collection("video_id_batch")[0]
    prediction_batch = tf.get_collection("predictions")[0]
    label_batch = tf.get_collection("labels")[0]

    saver = tf.train.Saver(tf.global_variables())

    inference_loop(video_id_batch, prediction_batch, label_batch,
                   saver, FLAGS.output_file)

def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  print("tensorflow version: %s" % tf.__version__)
  inference()


if __name__ == "__main__":
  app.run()

