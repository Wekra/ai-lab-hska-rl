import tensorflow as tf


class TensorBoardLogger:
    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, name, value, global_step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
        self.writer.add_summary(summary, global_step)

    def log_image(self, name, image, global_step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, image=image)])
        self.writer.add_summary(summary, global_step)