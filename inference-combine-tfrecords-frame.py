# Copyright 2016 Google Inc. All Rights Reserved.
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

"""Binary for combine model output and model input into one set of files."""

import os
import sys
import time
import numpy
import numpy as np

import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

import utils
import eval_util
import losses
import readers
import ensemble_level_models

FLAGS = flags.FLAGS

if __name__ == '__main__':
  flags.DEFINE_string("output_dir", "",
                      "The file to save the predictions to.")
  flags.DEFINE_string(
      "input_data_pattern", "",
      "File globs defining the input dataset in tensorflow.SequenceExample format.")
  flags.DEFINE_string(
      "prediction_data_pattern", "",
      "File globs defining the output dataset in tensorflow.SequenceExample format.")
  flags.DEFINE_string("input_feature_names", "mean_rgb,mean_audio", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("input_feature_sizes", "1024,128", "Length of the feature vectors.")
  flags.DEFINE_string("prediction_feature_names", "predictions", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("prediction_feature_sizes", "3862", "Length of the feature vectors.")
  flags.DEFINE_integer("batch_size", 256,
                       "How many examples to process per batch.")
  flags.DEFINE_integer("file_size", 4096,
                       "Number of frames per batch for DBoF.")
  flags.DEFINE_integer("file_num_mod", None,
                       "file_num % 3 == file_num_mod will be output.")

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
        capacity=4 * batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)

def build_graph(input_reader, input_data_pattern,
                prediction_reader, prediction_data_pattern,
                batch_size=256):
  """Creates the Tensorflow graph for evaluation.

  Args:
    all_readers: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    all_data_patterns: glob path to the evaluation data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
  """

  video_ids_batch, model_rgbs_batch, model_audios_batch, labels_batch, num_frames_batch = (
      get_input_data_tensors(
          input_reader,
          input_data_pattern,
          batch_size=batch_size))
  video_ids_batch2, model_predictions_batch, labels_batch2, unused_num_frames2 = (
      get_input_data_tensors(
          prediction_reader,
          prediction_data_pattern,
          batch_size=batch_size))

  tf.add_to_collection("video_ids_batch", video_ids_batch)
  tf.add_to_collection("labels_batch", tf.cast(labels_batch, tf.float32))
  tf.add_to_collection("rgbs_batch", model_rgbs_batch)
  tf.add_to_collection("audios_batch", model_audios_batch)
  tf.add_to_collection("predictions_batch", model_predictions_batch)
  tf.add_to_collection("num_frames_batch", num_frames_batch)


def inference_loop(video_ids_batch, labels_batch, rgbs_batch, audios_batch, predictions_batch, num_frames_batch,
                   output_dir, batch_size):

  with tf.Session() as sess:

    sess.run([tf.local_variables_initializer()])

    # Start the queue runners.
    fetches = [video_ids_batch, labels_batch, rgbs_batch, audios_batch, predictions_batch, num_frames_batch]
    coord = tf.train.Coordinator()
    start_time = time.time()

    video_ids = []
    video_labels = []
    video_rgbs = []
    video_audios = []
    video_predictions = []
    video_num_frames = []
    filenum = 0
    num_examples_processed = 0
    total_num_examples_processed = 0

    directory = FLAGS.output_dir
    if not os.path.exists(directory):
        os.makedirs(directory)

    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True,
            start=True))

      while not coord.should_stop():
        ids_val = None
        ids_val, labels_val, rgbs_val, audios_val, predictions_val, num_frames_val = sess.run(fetches)

        if filenum % 3 == FLAGS.file_num_mod or FLAGS.file_num_mod is None:
          video_ids.append(ids_val)
          video_labels.append(labels_val)
          video_rgbs.append(rgbs_val)
          video_audios.append(audios_val)
          video_predictions.append(predictions_val)
          video_num_frames.append(num_frames_val)

        num_examples_processed += len(ids_val)

        ids_shape = ids_val.shape[0]
        rgbs_shape = rgbs_val.shape[0]
        audios_shape = audios_val.shape[0]
        predictions_shape = predictions_val.shape[0]
        assert ids_shape == rgbs_shape == predictions_shape, "tensor ids(%d), rgbs(%d) and predictions(%d) should have equal rows" % (ids_shape, rgbs_shape, predictions_shape)

        ids_val = None

        now = time.time()
        logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))

        if num_examples_processed >= FLAGS.file_size:
          assert num_examples_processed==FLAGS.file_size, "num_examples_processed should be equal to %d"%FLAGS.file_size

          if filenum % 3 == FLAGS.file_num_mod or FLAGS.file_num_mod is None:
            video_ids = np.concatenate(video_ids, axis=0)
            video_labels = np.concatenate(video_labels, axis=0)
            video_rgbs = np.concatenate(video_rgbs, axis=0)
            video_audios = np.concatenate(video_audios, axis=0)
            video_predictions = np.concatenate(video_predictions, axis=0)
            video_num_frames = np.concatenate(video_num_frames, axis=0)
            write_to_record(video_ids, video_labels, video_rgbs, video_audios, video_predictions, video_num_frames, filenum, num_examples_processed)

          video_ids = []
          video_labels = []
          video_rgbs = []
          video_audios = []
          video_predictions = []
          video_num_frames = []
          filenum += 1
          total_num_examples_processed += num_examples_processed

          now = time.time()
          logging.info(str(num_examples_processed) + " examples written to file elapsed seconds: " + "{0:.2f}".format(now-start_time))
          num_examples_processed = 0

    except tf.errors.OutOfRangeError as e:
      if ids_val is not None:
        video_ids.append(ids_val)
        video_labels.append(labels_val)
        video_rgbs.append(rgbs_val)
        video_audios.append(audios_val)
        video_predictions.append(predictions_val)
        video_num_frames.append(num_frames_val)
        num_examples_processed += len(ids_val)

      if 0 < num_examples_processed <= FLAGS.file_size:
        if filenum % 3 == FLAGS.file_num_mod or FLAGS.file_num_mod is None:
          video_ids = np.concatenate(video_ids, axis=0)
          video_labels = np.concatenate(video_labels, axis=0)
          video_rgbs = np.concatenate(video_rgbs, axis=0)
          video_audios = np.concatenate(video_audios, axis=0)
          video_predictions = np.concatenate(video_predictions, axis=0)
          video_num_frames = np.concatenate(video_num_frames, axis=0)
          write_to_record(video_ids, video_labels, video_rgbs, video_audios, video_predictions, video_num_frames, filenum, num_examples_processed)

        total_num_examples_processed += num_examples_processed

        now = time.time()
        logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))
        num_examples_processed = 0

      logging.info("Done with inference. %d samples was written to %s" % (total_num_examples_processed, FLAGS.output_dir))
    except Exception as e:  # pylint: disable=broad-except
      logging.info("Unexpected exception: " + str(e))
    finally:
      coord.request_stop()

    coord.join(threads, stop_grace_period_secs=10)


def write_to_record(video_ids, video_labels, video_rgbs, video_audios, video_predictions, video_num_frames, filenum, num_examples_processed):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_dir + '/' + 'predictions-%04d.tfrecord' % filenum)
    for i in range(num_examples_processed):
        video_id = video_ids[i]
        video_label = np.nonzero(video_labels[i,:])[0]
        video_rgb = video_rgbs[i,:]
        video_audio = video_audios[i,:]
        video_prediction = video_predictions[i,:]
        video_num_frame = video_num_frames[i]
        example = get_output_feature(video_id, video_label, video_rgb, video_audio, video_prediction, video_num_frame)
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()

def get_output_feature(video_id, video_label, video_rgb, video_audio, video_prediction, video_num_frame):
    _bytes_feature_list = lambda x: tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))
    example = tf.train.SequenceExample(
        context = tf.train.Features(feature={
            "video_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_id])),
            "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=video_label)),
            "predictions": tf.train.Feature(float_list=tf.train.FloatList(value=video_prediction))
        }),
        feature_lists = tf.train.FeatureLists(feature_list={
            "rgb": tf.train.FeatureList(feature=map(_bytes_feature_list, video_rgb[:video_num_frame])),
            "audio": tf.train.FeatureList(feature=map(_bytes_feature_list, video_audio[:video_num_frame])),
        })
    )
    return example

def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    if FLAGS.input_data_pattern is "":
      raise IOError("'input_data_pattern' was not specified. " +
                     "Nothing to evaluate.")
    if FLAGS.prediction_data_pattern is "":
      raise IOError("'prediction_data_pattern' was not specified. " +
                     "Nothing to evaluate.")

    # convert feature_names and feature_sizes to lists of values
    input_feature_names, input_feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        FLAGS.input_feature_names, FLAGS.input_feature_sizes)
    input_reader = readers.EnsembleFrameReader(
        feature_names=input_feature_names, 
        feature_sizes=input_feature_sizes)

    prediction_feature_names, prediction_feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        FLAGS.prediction_feature_names, FLAGS.prediction_feature_sizes)
    prediction_reader = readers.EnsembleReader(
        feature_names=prediction_feature_names, 
        feature_sizes=prediction_feature_sizes)

    build_graph(
        input_reader=input_reader,
        prediction_reader=prediction_reader,
        input_data_pattern=FLAGS.input_data_pattern,
        prediction_data_pattern=FLAGS.prediction_data_pattern,
        batch_size=FLAGS.batch_size)

    logging.info("built evaluation graph")

    video_ids_batch = tf.get_collection("video_ids_batch")[0]
    labels_batch = tf.get_collection("labels_batch")[0]
    rgbs_batch = tf.get_collection("rgbs_batch")[0]
    audios_batch = tf.get_collection("audios_batch")[0]
    predictions_batch = tf.get_collection("predictions_batch")[0]
    num_frames_batch = tf.get_collection("num_frames_batch")[0]

    inference_loop(video_ids_batch, labels_batch, rgbs_batch, audios_batch, predictions_batch, num_frames_batch,
                   FLAGS.output_dir, FLAGS.batch_size)
  
if __name__ == "__main__":
  app.run()
