	��F!�!@��F!�!@!��F!�!@	�w���U!@�w���U!@!�w���U!@"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��F!�!@�',�r�?1�1uWv�@A=|�(B�?I�����T�?Y���x���?rEagerKernelExecute 0*	V-�g@2F
Iterator::Modelv��ť*�?!�C�K�J@)]����۩?1�|���:@:Preprocessing2U
Iterator::Model::ParallelMapV2��]��y�?!�
���L9@)��]��y�?1�
���L9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat>���d��?!h�sv�-3@)�G�)s�?1�����.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice��J�RН?!�и��.@)��J�RН?1�и��.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip׿�3�?!:�9�)�G@)��:���?1wɢaM @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap+�����?!Po� "�4@)���=�>�?1-�XL�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorIJzZ�|?!q�lL�@)IJzZ�|?1q�lL�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 8.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"�14.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t15.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�w���U!@Ir����5>@Qe�����N@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�',�r�?�',�r�?!�',�r�?      ��!       "	�1uWv�@�1uWv�@!�1uWv�@*      ��!       2	=|�(B�?=|�(B�?!=|�(B�?:	�����T�?�����T�?!�����T�?B      ��!       J	���x���?���x���?!���x���?R      ��!       Z	���x���?���x���?!���x���?b      ��!       JGPUY�w���U!@b qr����5>@ye�����N@