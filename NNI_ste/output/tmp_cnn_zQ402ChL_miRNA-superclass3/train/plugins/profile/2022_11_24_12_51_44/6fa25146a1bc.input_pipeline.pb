	�J����@�J����@!�J����@	���	�� @���	�� @!���	�� @"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�J����@R�o&��?1�x���%�?A}��b٥?I_'�eiW@YRF\ ��?rEagerKernelExecute 0*	D�l��d@2F
Iterator::Modelu�BY��?!3r��K@)/����?1�d��GD@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatn���+�?!�}��t�8@)#�	��?1g�'%��4@:Preprocessing2U
Iterator::Model::ParallelMapV2e��)1�?!�(6�ä.@)e��)1�?1�(6�ä.@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�u��=�?!�+�W�@)�u��=�?1�+�W�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate2��Yؓ?!��ݣ�#(@)|���s�?1[d�wq�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipQ�i>"�?!���F@)�YL��?1v�Z���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor3�FY��x?!������@)3�FY��x?1������@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap}�;l"3�?!�>�5�8,@)}�E�j?1�-�G�R @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 21.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�57.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���	�� @In7����S@QJ&�y2@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	R�o&��?R�o&��?!R�o&��?      ��!       "	�x���%�?�x���%�?!�x���%�?*      ��!       2	}��b٥?}��b٥?!}��b٥?:	_'�eiW@_'�eiW@!_'�eiW@B      ��!       J	RF\ ��?RF\ ��?!RF\ ��?R      ��!       Z	RF\ ��?RF\ ��?!RF\ ��?b      ��!       JGPUY���	�� @b qn7����S@yJ&�y2@