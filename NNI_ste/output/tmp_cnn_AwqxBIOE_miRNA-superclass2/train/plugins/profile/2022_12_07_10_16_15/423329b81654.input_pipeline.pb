	z�]�z� @z�]�z� @!z�]�z� @      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCz�]�z� @��N�?1��#ӡ@A{Nz��ړ?I*���O;@rEagerKernelExecute 0*	cX9�\c@2F
Iterator::ModelS�'�ݲ?!�����G@)гY��ڪ?1�Fx)<�@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��jH�c�?!�D��@@)z ���!�?1�NI�V�;@:Preprocessing2U
Iterator::Model::ParallelMapV2������?!8}
n+@)������?18}
n+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�I�5�o�?!�S'.J@)�I�5�o�?1�S'.J@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipfKVE�ɴ?!u�TA6J@)5�l�/�?1��Kn�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(���֓?!�9�:�)@)k��t=�?1lY7N�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��^
z?!tl���m@)��^
z?1tl���m@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap_�(�QG�?!b�ko7Z-@)�y�Cn�k?1���|Z@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�42.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�����F@Q\@&�K@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��N�?��N�?!��N�?      ��!       "	��#ӡ@��#ӡ@!��#ӡ@*      ��!       2	{Nz��ړ?{Nz��ړ?!{Nz��ړ?:	*���O;@*���O;@!*���O;@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�����F@y\@&�K@