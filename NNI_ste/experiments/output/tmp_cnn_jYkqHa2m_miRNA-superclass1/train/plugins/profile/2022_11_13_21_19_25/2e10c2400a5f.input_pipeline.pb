	����{�@����{�@!����{�@	�{W�N @�{W�N @!�{W�N @"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC����{�@�}�u�r�?1����	@IӽN�˲@Y�]�����?rEagerKernelExecute 0*	�G�zu@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��`�$��?!��)Z��M@)�K�����?1b���K@:Preprocessing2U
Iterator::Model::ParallelMapV2P��0{٦?!Ϳ��5�*@)P��0{٦?1Ϳ��5�*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�2SZK�?!=��j.,@){���`�?1�L�x�'@:Preprocessing2F
Iterator::Model.��Hٲ?!2���,�5@)V-��?1����#9!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��8Q��?!�BH@)��8Q��?1�BH@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2t���?!t�Uƴ�S@)�4�\���?1�kB�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���͎T?!6��+@)���͎T?16��+@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaplxz�,C�?!7����vN@)�g�m?1�h�9*�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 2.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�43.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�{W�N @I���~��E@Qu�^DK@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�}�u�r�?�}�u�r�?!�}�u�r�?      ��!       "	����	@����	@!����	@*      ��!       2      ��!       :	ӽN�˲@ӽN�˲@!ӽN�˲@B      ��!       J	�]�����?�]�����?!�]�����?R      ��!       Z	�]�����?�]�����?!�]�����?b      ��!       JGPUY�{W�N @b q���~��E@yu�^DK@