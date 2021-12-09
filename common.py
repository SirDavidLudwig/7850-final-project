from multiprocessing import Array, Process, Queue, Value
import numpy as np
import os
import queue
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import settransformer as sf
import shelve
import time

def strategy(device=None, multi_gpu=False):
    """
    Create the distribution strategy for GPU(s).
    """
    
    device = device.upper() if device is not None else "CPU:0"
    
    # If we want multiple GPUs, return the mirrored strategy
    if multi_gpu:
        return tf.distribute.MirroredStrategy()

    # Fetch the required devices
    devices = tf.config.get_visible_devices("CPU")
    devices += [d for d in tf.config.get_visible_devices("GPU")
                    if d.name.endswith(device)]
    
    # Hide any unwanted devices
    tf.config.set_visible_devices(devices)
    return tf.distribute.OneDeviceStrategy(device)


class Benchmark:
    def __init__(self, label = "Total time"):
        self.label = label
        self.time = 0
        self.cp_time = 0
        
    def checkpoint(self, message):
        t = time.time()
        print(f"{message}; {t - self.cp_time} seconds")
        self.cp_time = t
    
    def __enter__(self):
        self.time = time.time()
        self.cp_time = self.time
        return self
        
    def __exit__(self, type, value, traceback):
        print(f"{self.label}: {time.time() - self.time} seconds.")
        return self

    
class MultiprocessMnistGenerator:
    def __init__(self, batch_size=32, num_points=500, buffer_size=5, num_workers=1, threshold=50, test=False, labels=None):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.num_points = num_points
        self.is_running = Value('b', False)
        self.num_workers = num_workers
        self.workers = []
        self.digits = self.load_data(threshold, test, labels)
        self.batches_x = np.frombuffer(Array('f', buffer_size*batch_size*num_points*2, lock=False), dtype=np.float32) \
                           .reshape((buffer_size, batch_size, num_points, 2))
        self.batches_y = np.frombuffer(Array('i', buffer_size*batch_size, lock=False), dtype=np.int32) \
                           .reshape((buffer_size, batch_size))
        self.batches = (self.batches_x, self.batches_y)
        self.ready_batches = Queue(buffer_size)
        self.stale_batches = Queue(buffer_size)
        self.current_batch = 0
        
        for i in range(1, buffer_size):
            self.stale_batches.put(i)
        
        
    def load_data(self, threshold, test, labels):
        """
        Loads information into shared memory
        """
        # Load MNIST
        training, testing = keras.datasets.mnist.load_data()
        x, y = testing if test else training
        
        if labels is not None:
            valid_y = np.zeros_like(y, dtype=np.bool)
            for label in labels:
                valid_y = np.logical_or(valid_y, y == label)
            indices = np.where(valid_y)
            x = x[indices]
            y = y[indices]
        
        # Extract valid pixels for points
        img_ids, y_pixels, x_pixels = np.nonzero(x > threshold)
        
        # Create shared pixel array
        pixel_shared_arr = Array('f', 2*len(img_ids), lock=False)
        pixels = np.frombuffer(pixel_shared_arr, dtype=np.float32).reshape((-1, 2))
        pixels[:] = np.column_stack((x_pixels, 28 - y_pixels))
        
        # Standardize the pixels
        mean = np.mean(pixels, axis=0)
        std = np.std(pixels, axis=0)
        pixels[:] = (pixels - mean) / std
        
        indices = np.frombuffer(Array('i', len(x), lock=False), dtype=np.int32)
        indices[:], pixel_counts = np.unique(img_ids, return_counts=True)
        pixel_offsets = np.concatenate([[0], np.cumsum(pixel_counts)])
        
        return indices, pixels, pixel_offsets, std, y
        
    def start(self):
        if self.is_running.value:
            raise Exception("Workers are already running")
        args = (
            self.is_running,
            self.batches,
            self.stale_batches,
            self.ready_batches,
            self.digits,
            self.batch_size,
            self.num_points)
        self.is_running.value = True
        for _ in range(self.num_workers):
            worker = Process(target=MultiprocessMnistGenerator.worker, args=args)
            worker.start()
            self.workers.append(worker)
        
    def stop(self):
        if not self.is_running.value:
            raise Exception("Workers are already stopped")
        self.is_running.value = False
        for worker in self.workers:
            worker.join()
        self.workers = []
            
    def terminate(self):
        self.is_running.value = False
        for worker in self.workers:
            worker.terminate()
        self.workers = []
        
    def generator(self):
        while True:
            yield next(self)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        # Mark the current batch as stale
        self.stale_batches.put(self.current_batch)
        self.current_batch = self.ready_batches.get()
        return self.batches_x[self.current_batch], self.batches_y[self.current_batch]
        
    @staticmethod
    def generate_batch(batches, batch_id, digits, batch_size, num_points):
        batches_x, batches_y = batches
        x, y = batches_x[batch_id], batches_y[batch_id]
        
        indices, pixels, offsets, std, labels = digits
        indices = np.random.choice(indices, size=batch_size, replace=False)
        y[:] = labels[indices]
        
        for i, digit in enumerate(indices):
            px = pixels[offsets[digit]:offsets[digit+1]]
            x[i] = px[np.random.randint(0, len(px), size=num_points)]
            x[i] += np.random.uniform(0.0, 1/std, size=x[i].shape)
            
    @staticmethod
    def worker(is_running, batches, stale_batches, ready_batches, digits, batch_size, num_points):
        while is_running.value and os.getppid() != 1:
            try:
                batch_id = stale_batches.get(timeout=1.0)
                MultiprocessMnistGenerator.generate_batch(batches, batch_id, digits, batch_size, num_points)
                ready_batches.put(batch_id)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                continue

                
class MultiprocessPairedMnistGenerator:
    def __init__(self, label_a, label_b, batch_size=32, num_points=500, buffer_size=5, num_workers=1, threshold=50, test=False):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.num_points = num_points
        self.is_running = Value('b', False)
        self.num_workers = num_workers
        self.workers = []
        self.digits = self.load_data(label_a, label_b, threshold, test)
        self.batches_a = np.frombuffer(Array('f', buffer_size*batch_size*num_points*2, lock=False), dtype=np.float32) \
                           .reshape((buffer_size, batch_size, num_points, 2))
        self.batches_b = np.frombuffer(Array('f', buffer_size*batch_size*num_points*2, lock=False), dtype=np.float32) \
                           .reshape((buffer_size, batch_size, num_points, 2))
        self.batches = (self.batches_a, self.batches_b)
        self.ready_batches = Queue(buffer_size)
        self.stale_batches = Queue(buffer_size)
        self.current_batch = 0
        
        for i in range(1, buffer_size):
            self.stale_batches.put(i)
            
        
    def load_data(self, label_a, label_b, threshold, test):
        """
        Loads information into shared memory
        """
        # Load MNIST
        training, testing = keras.datasets.mnist.load_data()
        x, y = testing if test else training
        
        # Find indices of desired digits
        a_indices = np.where(y == label_a)[0]
        b_indices = np.where(y == label_b)[0]
        num_indices = min(len(a_indices), len(b_indices))
        a_indices = a_indices[:num_indices]
        b_indices = b_indices[:num_indices]
        ab_indices = np.concatenate((a_indices, b_indices))
        
        # Trim x and y
        x = x[ab_indices]
        y = y[ab_indices]
        
        # Extract valid pixels for points
        img_ids, y_pixels, x_pixels = np.nonzero(x > threshold)
        
        # Create shared pixel array
        pixel_shared_arr = Array('f', 2*len(img_ids), lock=False)
        pixels = np.frombuffer(pixel_shared_arr, dtype=np.float32).reshape((-1, 2))
        pixels[:] = np.column_stack((x_pixels, 28 - y_pixels))
        
        # Standardize the pixels
        mean = np.mean(pixels, axis=0)
        std = np.std(pixels, axis=0)
        pixels[:] = (pixels - mean) / std
        
        a_indices = np.frombuffer(Array('i', num_indices, lock=False), dtype=np.int32)
        b_indices = np.frombuffer(Array('i', num_indices, lock=False), dtype=np.int32)
        a_indices[:] = np.where(y == label_a)[0]
        b_indices[:] = np.where(y == label_b)[0]
        pixel_counts = np.unique(img_ids, return_counts=True)[1]
        pixel_offsets = np.concatenate([[0], np.cumsum(pixel_counts)])
        
        return a_indices, b_indices, pixels, pixel_offsets, std
    
        
    def start(self):
        if self.is_running.value:
            raise Exception("Workers are already running")
        args = (
            self.is_running,
            self.batches,
            self.stale_batches,
            self.ready_batches,
            self.digits,
            self.batch_size,
            self.num_points)
        self.is_running.value = True
        for _ in range(self.num_workers):
            worker = Process(target=MultiprocessPairedMnistGenerator.worker, args=args)
            worker.start()
            self.workers.append(worker)
        
        
    def stop(self):
        if not self.is_running.value:
            raise Exception("Workers are already stopped")
        self.is_running.value = False
        for worker in self.workers:
            worker.join()
        self.workers = []
            
            
    def terminate(self):
        self.is_running.value = False
        for worker in self.workers:
            worker.terminate()
        self.workers = []
        
        
    def generator(self):
        while True:
            yield next(self)
        
        
    def __iter__(self):
        return self
        
        
    def __next__(self):
        # Mark the current batch as stale
        self.stale_batches.put(self.current_batch)
        self.current_batch = self.ready_batches.get()
        return self.batches_a[self.current_batch], self.batches_b[self.current_batch]
        
        
    @staticmethod
    def generate_batch(batches, batch_id, digits, batch_size, num_points):
        batches_a, batches_b = batches
        a, b = batches_a[batch_id], batches_b[batch_id]
        
        a_indices, b_indices, pixels, offsets, std = digits
        indices = np.random.choice(np.arange(len(a_indices)), size=batch_size, replace=False)
        a_indices = a_indices[indices]
        b_indices = b_indices[indices]
        
        for batch, indices in [(a, a_indices), (b, b_indices)]:
            for i, digit in enumerate(indices):
                px = pixels[offsets[digit]:offsets[digit+1]]
                batch[i] = px[np.random.randint(0, len(px), size=num_points)]
                batch[i] += np.random.uniform(0.0, 1/std, size=batch[i].shape)
            
            
    @staticmethod
    def worker(is_running, batches, stale_batches, ready_batches, digits, batch_size, num_points):
        while is_running.value and os.getppid() != 1:
            try:
                batch_id = stale_batches.get(timeout=1.0)
                MultiprocessPairedMnistGenerator.generate_batch(batches, batch_id, digits, batch_size, num_points)
                ready_batches.put(batch_id)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                continue
                

class MultiprocessDnaSampler:
    def __init__(self, files, sequence_length=150, kmers=1, subsample_size=1000, batch_size=32, buffer_size=5, num_workers=1):
        self.files = files
        self.sequence_length = sequence_length
        self.kmers = kmers
        self.kmer_sequence_length = int(sequence_length / kmers)
        self.subsample_size = subsample_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        self.is_running = Value('b', False)
        self.num_workers = num_workers
        self.workers = []
        self.batches = np.frombuffer(Array('i', buffer_size*batch_size*subsample_size*self.kmer_sequence_length, lock=False), dtype=np.int32) \
                        .reshape((buffer_size, batch_size, subsample_size, self.kmer_sequence_length))
        self.ready_batches = Queue(buffer_size)
        self.stale_batches = Queue(buffer_size)
        self.current_batch = 0
        
        for i in range(1, buffer_size):
            self.stale_batches.put(i)
        
    def start(self):
        if self.is_running.value:
            raise Exception("Workers are already running")
        args = (
            self.is_running,
            self.batches,
            self.stale_batches,
            self.ready_batches,
            self.files,
            self.batch_size,
            self.sequence_length,
            self.kmers,
            self.subsample_size)
        self.is_running.value = True
        for _ in range(self.num_workers):
            worker = Process(target=MultiprocessDnaSampler.worker, args=args)
            worker.start()
            self.workers.append(worker)
        
    def stop(self):
        if not self.is_running.value:
            raise Exception("Workers are already stopped")
        self.is_running.value = False
        for worker in self.workers:
            worker.join()
        self.workers = []
            
    def terminate(self):
        self.is_running.value = False
        for worker in self.workers:
            worker.terminate()
        self.workers = []
        
    def generator(self):
        while True:
            yield next(self)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        # Mark the current batch as stale
        self.stale_batches.put(self.current_batch)
        self.current_batch = self.ready_batches.get()
        return self.batches[self.current_batch]
    
    @staticmethod
    def augment(sample, sequence_length):
        offset = np.random.randint(sample.shape[0] - sequence_length + 1)
        sample = sample[offset:sequence_length+offset]
        return sample
        
    @staticmethod
    def subsample(stores, store_lengths, subsample_size, sequence_length):
        sample_index = np.random.randint(len(stores))
        num_samples = store_lengths[sample_index]
        sequences = np.random.choice(np.arange(num_samples), subsample_size, replace=False)
        return np.array([MultiprocessDnaSampler.augment(stores[sample_index][str(s)][0], sequence_length) for s in sequences])
    
    @staticmethod
    def generate_batch(stores, store_lengths, batches, batch_id, batch_size, subsample_size, sequence_length, kmers, kmer_powers):
        batch = np.array([MultiprocessDnaSampler.subsample(stores, store_lengths, subsample_size, sequence_length) for s in range(batch_size)])
        batches[batch_id] = np.sum(batch.reshape((batch_size, subsample_size, -1, kmers)) * kmer_powers, axis=3)
            
    @staticmethod
    def worker(is_running, batches, stale_batches, ready_batches, files, batch_size, sequence_length, kmers, subsample_size):
        stores = [shelve.open(f) for f in files] # not thread safe, so each worker has own instance
        store_lengths = [s["length"] for s in stores]
        kmer_sequence_length = int(sequence_length / kmers)
        kmer_powers = np.full(kmers, 5)**np.arange(kmers - 1, -1, -1)
        
        while is_running.value and os.getppid() != 1:
            try:
                batch_id = stale_batches.get(timeout=1.0)
                MultiprocessDnaSampler.generate_batch(stores, store_lengths, batches, batch_id, batch_size, subsample_size, sequence_length, kmers, kmer_powers)
                ready_batches.put(batch_id)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                continue
                
                
# class ConditionedISAB(keras.layers.Layer):
#     def __init__(self, embed_dim, dim_cond, num_heads, num_anchors):
#         super(ConditionedISAB, self).__init__()
#         self.embed_dim = embed_dim
#         self.num_anchors = num_anchors
#         self.mab1 = sf.MAB(embed_dim, num_heads)
#         self.mab2 = sf.MAB(embed_dim, num_heads)
#         self.anchor_predict = keras.models.Sequential([
#             keras.layers.Dense(
#                 num_anchors*embed_dim,
#                 input_shape=(dim_cond,),
#                 activation="gelu"),
#             tfa.layers.SpectralNormalization(
#                 keras.layers.Dense(num_anchors*embed_dim)),
#             keras.layers.Reshape((num_anchors, embed_dim))
#         ])
        
#     def call(self, inp):
#         inducing_points = self.anchor_predict(inp[1])
#         h = self.mab1(inducing_points, inp[0])
#         return self.mab2(inp[0], h)

class ConditionedISAB(keras.layers.Layer):
    def __init__(self, embed_dim, dim_cond, num_heads, num_anchors):
        super(ConditionedISAB, self).__init__()
        self.embed_dim = embed_dim
        self.num_anchors = num_anchors
        self.mab1 = sf.MAB(embed_dim, num_heads)
        self.mab2 = sf.MAB(embed_dim, num_heads)
        self.anchor_predict = keras.models.Sequential([
            keras.layers.Dense(
                2*dim_cond,
                input_shape=(dim_cond,),
                activation="gelu"),
            tfa.layers.SpectralNormalization(
                keras.layers.Dense(num_anchors*embed_dim)),
            keras.layers.Reshape((num_anchors, embed_dim))
        ])
        
    def call(self, inp):
        inducing_points = self.anchor_predict(inp[1])
        h = self.mab1(inducing_points, inp[0])
        return self.mab2(inp[0], h)
    
    
class SampleSet(keras.layers.Layer):
    def __init__(self, max_set_size, embed_dim):
        super(SampleSet, self).__init__()
        self.max_set_size = max_set_size
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.mu = self.add_weight(
            shape=(self.max_set_size, self.embed_dim),
            initializer="random_normal",
            trainable=True,
            name="mu")
        self.sigma = self.add_weight(
            shape=(self.max_set_size, self.embed_dim),
            initializer="random_normal",
            trainable=True,
            name="sigma")
        
    def call(self, n):
        batch_size = tf.shape(n)[0]
        n = tf.squeeze(tf.cast(n[0], dtype=tf.int32)) # all n should be the same, take one
#         n = self.max_set_size
        mean = self.mu
        variance = tf.square(self.sigma)
        
        # Sample a random initial set of max size
        initial_set = tf.random.normal((batch_size, self.max_set_size, self.embed_dim), mean, variance)

        # Pick random indices without replacement
        _, random_indices = tf.nn.top_k(tf.random.uniform(shape=(batch_size, self.max_set_size)), n)        
        batch_indices = tf.reshape(tf.repeat(tf.range(batch_size), n), (-1, n))
        indices = tf.stack([batch_indices, random_indices], axis=2)
        
        # Sample the set
        sampled_set = tf.gather_nd(initial_set, indices)
        
        return sampled_set
    
class BaseTransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(BaseTransformerBlock, self).__init__()
        self.ffn = keras.Sequential(
            [keras.layers.Dense(ff_dim, activation="gelu"),
             keras.layers.Dense(embed_dim),]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.att = self.create_attention_layer(embed_dim, num_heads)
        
    def create_attention_layer(self, embed_dim, num_heads):
        raise NotImplemented()
        
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    

class TransformerBlock(BaseTransformerBlock):
    def create_attention_layer(self, embed_dim, num_heads):
        return keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    
    
class FastformerBlock(BaseTransformerBlock):        
    def create_attention_layer(self, embed_dim, num_heads):
        return Fastformer(embed_dim, num_heads)
    
    
class FixedPositionEmbedding(keras.layers.Layer):
    def __init__(self, length, embed_dim):
        super(FixedPositionEmbedding, self).__init__()
        self.length = length
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.positions = self.add_weight(
            shape=(self.length, self.embed_dim),
            initializer="uniform",
            trainable=True)
        
    def call(self, x):
        return x + self.positions
