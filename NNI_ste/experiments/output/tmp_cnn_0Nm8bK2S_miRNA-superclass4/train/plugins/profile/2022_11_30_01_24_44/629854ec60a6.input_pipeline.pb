	??I؟!@??I؟!@!??I؟!@      ??!       "{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??I؟!@?(]????1?I'Le@Ic%?YIK@rEagerKernelExecute 0*	x?&1>u@2F
Iterator::Model6׆?q??!?Q{d?R@)B?f??j??1O?z???P@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??U?Z??!`??? W)@)"?^F?ܢ?1%?֢??%@:Preprocessing2U
Iterator::Model::ParallelMapV2Sy;?i???!]f?W?@)Sy;?i???1]f?W?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???"???!????Ȱ@)???"???1????Ȱ@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?"M?<??!/?n?g8@)?????=??1,?4e'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?w(
????!??o?if @)HĔH????1x??8@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?f??I}y?!?!#	?K??)?f??I}y?1?!#	?K??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??'?8??!4^?>??!@)??$?pte?1&V?v????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?28.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?ca[<?>@Q?'?pMQ@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?(]?????(]????!?(]????      ??!       "	?I'Le@?I'Le@!?I'Le@*      ??!       2      ??!       :	c%?YIK@c%?YIK@!c%?YIK@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?ca[<?>@y?'?pMQ@