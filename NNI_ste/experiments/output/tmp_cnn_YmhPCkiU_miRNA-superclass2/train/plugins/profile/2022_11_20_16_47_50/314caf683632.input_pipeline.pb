	Z�!�[M@Z�!�[M@!Z�!�[M@      ��!       "�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCZ�!�[M@��ӹ���?1ѓ2�!@A2���j�?I��@rEagerKernelExecute 0*	fffffre@2F
Iterator::Model��N@a�?!�3�!�J@)���?1z]�{�A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��`�.�?!������8@),���c�?1�R��S�4@:Preprocessing2U
Iterator::Model::ParallelMapV2���U��?!K��K�1@)���U��?1K��K�1@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��.�u��?!$]���z@)��.�u��?1$]���z@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�vj.7�?!ګ/^m�-@)E�a���?1���2��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip%��C��?!`���bG@)S!����?1\�0�_@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�V$&��{?!���'�@)�V$&��{?1���'�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���>嘜?!p��o�F0@)�I�pd?15����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 21.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�45.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIѫ[���P@Q^�H��H@@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��ӹ���?��ӹ���?!��ӹ���?      ��!       "	ѓ2�!@ѓ2�!@!ѓ2�!@*      ��!       2	2���j�?2���j�?!2���j�?:	��@��@!��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qѫ[���P@y^�H��H@@