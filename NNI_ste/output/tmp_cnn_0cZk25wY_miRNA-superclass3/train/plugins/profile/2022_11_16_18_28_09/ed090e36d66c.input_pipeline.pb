	���=@���=@!���=@	��=g�@��=g�@!��=g�@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL���=@�l�Ԃ�?1�����@Acb�qm��?I���=�@Y�K�1�=�?rEagerKernelExecute 0*	���Ssf@2F
Iterator::Model�V%�}��?!�GV�RsG@)��Y�N�?1L8lA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�}��!�?!W��@8@)�"�tuǢ?1�݄��k4@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlices�}��؟?!�K��Q1@)s�}��؟?1�K��Q1@:Preprocessing2U
Iterator::Model::ParallelMapV2�-s�,�?!B��9(@)�-s�,�?1B��9(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateDkE��?!`��qQ�6@)�_���܃?1��Ѱ?�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip׽�	j�?!a��X��J@)a��pɁ?18C
��W@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor֪]�z?!ʹl�*@)֪]�z?1ʹl�*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�����?�?!�KJ�-28@)�T�:�e?1��6í�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 16.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�32.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��=g�@I`,�6SfH@QP��r��G@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�l�Ԃ�?�l�Ԃ�?!�l�Ԃ�?      ��!       "	�����@�����@!�����@*      ��!       2	cb�qm��?cb�qm��?!cb�qm��?:	���=�@���=�@!���=�@B      ��!       J	�K�1�=�?�K�1�=�?!�K�1�=�?R      ��!       Z	�K�1�=�?�K�1�=�?!�K�1�=�?b      ��!       JGPUY��=g�@b q`,�6SfH@yP��r��G@