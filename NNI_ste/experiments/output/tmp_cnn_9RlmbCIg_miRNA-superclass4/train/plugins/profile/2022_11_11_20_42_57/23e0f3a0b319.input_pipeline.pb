	 �����@ �����@! �����@	%y���x@%y���x@!%y���x@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL �����@`�;�I�?1����	@A����w�?Ic~nh�@Y�cϞ���?rEagerKernelExecute 0*	��� �_@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�n��?!p_�
G�B@)G�tF^�?1�Da"�A@:Preprocessing2F
Iterator::Model�v���?!ϩ���[D@)a5��6ƞ?1�a� �-8@:Preprocessing2U
Iterator::Model::ParallelMapV2s�V{��?!��>;�0@)s�V{��?1��>;�0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�A{��?!1���{8,@)ܼqR���?1h�Ť��#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipfM,�ݲ?!1Vb`m�M@)�Y�b+hz?1A��ow�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor~b��u?!�aݩ65@)~b��u?1�aݩ65@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�0(�hr�?!=5t_�C@)6l��ge?1�\��� @:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor$
-���`?!�,�(<�?)$
-���`?1�,�(<�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicev�A]�PV?!�my�l��?)v�A]�PV?1�my�l��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 16.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�35.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9&y���x@Iv�KU�I@Qmr|jSF@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	`�;�I�?`�;�I�?!`�;�I�?      ��!       "	����	@����	@!����	@*      ��!       2	����w�?����w�?!����w�?:	c~nh�@c~nh�@!c~nh�@B      ��!       J	�cϞ���?�cϞ���?!�cϞ���?R      ��!       Z	�cϞ���?�cϞ���?!�cϞ���?b      ��!       JGPUY&y���x@b qv�KU�I@ymr|jSF@