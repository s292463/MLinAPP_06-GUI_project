	��TlL-@��TlL-@!��TlL-@	47Y�Z�?47Y�Z�?!47Y�Z�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL��TlL-@�/����?1GXT���?ARE�*k�?I�67�'<'@Y! _B��?rEagerKernelExecute 0*	;�O��^a@2F
Iterator::Model�f��Ӭ?!?DaKIBD@)�=�N��?1�	Z��59@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��s]�?!��A�g8@)�fI-��?1G'ѳ4@:Preprocessing2U
Iterator::Model::ParallelMapV2u�i�ȕ?!���荝.@)u�i�ȕ?1���荝.@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	��Ln�?!@E	���,@)	��Ln�?1@E	���,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�2��(�?!������M@)"� ˂��?1o�)&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate6[y���?!��>j6@)Z�wg�?1ب�� @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensoru�Rz��x?!~���}H@)u�Rz��x?1~���}H@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapb�*�3�?!�6��7@)z�ަ?�a?1��'1F�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 8.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�79.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no947Y�Z�?Id�j��6V@Qn��YM%@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�/����?�/����?!�/����?      ��!       "	GXT���?GXT���?!GXT���?*      ��!       2	RE�*k�?RE�*k�?!RE�*k�?:	�67�'<'@�67�'<'@!�67�'<'@B      ��!       J	! _B��?! _B��?!! _B��?R      ��!       Z	! _B��?! _B��?!! _B��?b      ��!       JGPUY47Y�Z�?b qd�j��6V@yn��YM%@