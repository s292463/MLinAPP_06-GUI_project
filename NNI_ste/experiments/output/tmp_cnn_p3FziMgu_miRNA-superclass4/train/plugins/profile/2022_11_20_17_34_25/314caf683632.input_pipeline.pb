	�Ljh�!@�Ljh�!@!�Ljh�!@	A�Q���?A�Q���?!A�Q���?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL�Ljh�!@r��7y�?1�=b��@AH2�w��?I�� �?Yio���T�?rEagerKernelExecute 0*	�G�z|d@2F
Iterator::Model@�t�_��?!%��B�J@)�O�Y��?1���� C@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����5˥?!]�ԍ��9@)�,�Yf�?1�ل[9�5@:Preprocessing2U
Iterator::Model::ParallelMapV2R��񘁚?!�M<��/@)R��񘁚?1�M<��/@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���+,�?!�ق۟�@)���+,�?1�ق۟�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��Ϲە?!��3�*@)R�>�G��?1&~�,�J@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip)��Rb�?!�8(��G@)ס���Á?1m��+@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor75�|�}?!�F@ɼ�@)75�|�}?1�F@ɼ�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�P���?!}��Z�-@)�O:�`�i?1.��E��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 14.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�19.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9B�Q���?I陪I�YA@Q������O@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	r��7y�?r��7y�?!r��7y�?      ��!       "	�=b��@�=b��@!�=b��@*      ��!       2	H2�w��?H2�w��?!H2�w��?:	�� �?�� �?!�� �?B      ��!       J	io���T�?io���T�?!io���T�?R      ��!       Z	io���T�?io���T�?!io���T�?b      ��!       JGPUYB�Q���?b q陪I�YA@y������O@