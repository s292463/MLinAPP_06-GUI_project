	;�I/@;�I/@!;�I/@	���nζ@���nζ@!���nζ@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL;�I/@� 3���?1��?�Ŋ@Azq�ř?I��ɍ"� @Y��$\��?rEagerKernelExecute 0*	j�t�d@2F
Iterator::Model����%�?!��/oF@))�� l�?1��.ڽ=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���5��?!����?@@)e�u7�?1MݘF<@:Preprocessing2U
Iterator::Model::ParallelMapV2u���?!�=�^�,@)u���?1�=�^�,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice%�/�?!�����Z%@)%�/�?1�����Z%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��8d�?!@,�А�K@)�8�j�3�?1<��4b@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateJ�O�c�?!������/@)d �.���?1��9ʻ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_��W�{?!h	A܀�@)_��W�{?1h	A܀�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�fHū�?!}��Hu1@)���o
+e?1��r��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"�33.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t26.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9���nζ@I��Sv1�M@QT�ڻ�{A@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	� 3���?� 3���?!� 3���?      ��!       "	��?�Ŋ@��?�Ŋ@!��?�Ŋ@*      ��!       2	zq�ř?zq�ř?!zq�ř?:	��ɍ"� @��ɍ"� @!��ɍ"� @B      ��!       J	��$\��?��$\��?!��$\��?R      ��!       Z	��$\��?��$\��?!��$\��?b      ��!       JGPUY���nζ@b q��Sv1�M@yT�ڻ�{A@