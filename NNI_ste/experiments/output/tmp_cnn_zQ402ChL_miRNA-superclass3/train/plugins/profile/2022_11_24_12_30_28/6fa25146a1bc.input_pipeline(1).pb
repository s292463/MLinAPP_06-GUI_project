	u??&?@u??&?@!u??&?@	.@?/V\5@.@?/V\5@!.@?/V\5@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLu??&?@??y???1?-;?????A?vLݕ]??I}???e@Y?	????rEagerKernelExecute 0*	6^?Is@2U
Iterator::Model::ParallelMapV2p#e?????!? ?4c[M@)p#e?????1? ?4c[M@:Preprocessing2F
Iterator::Model4/??w??!??BƥR@)/?HM???1i??C??/@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeataE|??!x??e?N,@)???????1u=?rH?&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?D?$]3??!=?e>e-@)?D?$]3??1=?e>e-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateN	?I????!??a+MA@)??qn???1.????'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?\???ʳ?!?ߠ??h9@)???????1W?DD{?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor2??8*7??!?1?G@)2??8*7??1?1?G@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???
E??!蝲? @)????e?1Q*??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 21.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.high"?29.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*high2t24.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9.@?/V\5@I?2??w1K@Q?Z鞺@8@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??y?????y???!??y???      ??!       "	?-;??????-;?????!?-;?????*      ??!       2	?vLݕ]???vLݕ]??!?vLݕ]??:	}???e@}???e@!}???e@B      ??!       J	?	?????	????!?	????R      ??!       Z	?	?????	????!?	????b      ??!       JGPUY.@?/V\5@b q?2??w1K@y?Z鞺@8@