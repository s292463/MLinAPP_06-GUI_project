	����$�!@����$�!@!����$�!@	?�iz0� @?�iz0� @!?�iz0� @"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL����$�!@3Q���,�?1�#�G+@A���i��?I�kA�?Y+��O8��?rEagerKernelExecute 0*	�I+d@2F
Iterator::Modelwg��͵?!�5��F�J@)�Gp#e��?1�����*D@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatLR�b��?!�5K/�o;@)erjg�ڢ?1V�P[��6@:Preprocessing2U
Iterator::Model::ParallelMapV2��M��?!u�v�_�)@)��M��?1u�v�_�)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�i�L�?!wT��g@)�i�L�?1wT��g@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�`�d7�?!�p~�lG@)�_w��ă?1���rH@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateX��C��?!0���>'@)>����}?1��OKQ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;�vٯ;}?!���OK�@);�vٯ;}?1���OK�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapI.�!���?![״ag�*@)�7k�*g?1R���=�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 14.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�22.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?�iz0� @I@x�!B@Q\G���N@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	3Q���,�?3Q���,�?!3Q���,�?      ��!       "	�#�G+@�#�G+@!�#�G+@*      ��!       2	���i��?���i��?!���i��?:	�kA�?�kA�?!�kA�?B      ��!       J	+��O8��?+��O8��?!+��O8��?R      ��!       Z	+��O8��?+��O8��?!+��O8��?b      ��!       JGPUY?�iz0� @b q@x�!B@y\G���N@