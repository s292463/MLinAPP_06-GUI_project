	?*?MFU @?*?MFU @!?*?MFU @	Z????@Z????@!Z????@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL?*?MFU @Zg|_\???17߈?Y?@A)!XU/???I}???e @Y)[$?F??rEagerKernelExecute 0*	?v???c@2F
Iterator::Model;?zj???!?u?!%+F@)?LM?7???1N^0??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?e??S9??!?P??3:@)?n/i?֡?1ʘ??
6@:Preprocessing2U
Iterator::Model::ParallelMapV2wۅ?:???!?'?^)@)wۅ?:???1?'?^)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip/ܹ0ҋ??!????K@)??d?`T??1=?? ^?&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice7???0??!}͵??t&@)7???0??1}͵??t&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate0,?-X??!??5??B0@)????aN??1??j+? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???;{?!???R??@)???;{?1???R??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapz?9[@h??!??#??&2@)O??唀h?1?~????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?25.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9[????@I<?OO>?D@Q><?1?K@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Zg|_\???Zg|_\???!Zg|_\???      ??!       "	7߈?Y?@7߈?Y?@!7߈?Y?@*      ??!       2	)!XU/???)!XU/???!)!XU/???:	}???e @}???e @!}???e @B      ??!       J	)[$?F??)[$?F??!)[$?F??R      ??!       Z	)[$?F??)[$?F??!)[$?F??b      ??!       JGPUY[????@b q<?OO>?D@y><?1?K@