	~p>u�@~p>u�@!~p>u�@      ��!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:~p>u�@#��<�?1ѓ2��@I����� @rEagerKernelExecute 0*	2�ZǦ@2F
Iterator::Model�i�WV�@!�	\X@)���{@1A�X@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeati8en��?!3+����?)'�Ҩ��?1�����?:Preprocessing2U
Iterator::Model::ParallelMapV2MK��F>�?!ԛ�?��?)MK��F>�?1ԛ�?��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����ވ?!�`���?)����ވ?1�`���?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����>�?!������?)��Y.��?15N���?:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�7N
��?!�I��^~@)$D���?1_����b�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�K⬈z?!����Vp�?)�K⬈z?1����Vp�?:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�[��M�?!��E��?)XWj1xh?1��r�9�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�29.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�A���?@Q��Y� >Q@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	#��<�?#��<�?!#��<�?      ��!       "	ѓ2��@ѓ2��@!ѓ2��@*      ��!       2      ��!       :	����� @����� @!����� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�A���?@y��Y� >Q@