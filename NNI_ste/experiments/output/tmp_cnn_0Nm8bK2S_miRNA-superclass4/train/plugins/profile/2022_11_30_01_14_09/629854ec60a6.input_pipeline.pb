	3?68(@3?68(@!3?68(@	??di??????di????!??di????"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL3?68(@????K???1??!???#@AS<.?ED??I7???* @YO ????rEagerKernelExecute 0*	?MbXe@2F
Iterator::Model??G????!/??PH@)⬈?????1?`?^$>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat[%X????!w??X?@)ap??/??1?6)?,?:@:Preprocessing2U
Iterator::Model::ParallelMapV2ӽN??Ҟ?!??]?C?1@)ӽN??Ҟ?1??]?C?1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?? x|??!? ?K?@)?? x|??1? ?K?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?.l?V^??!?\?#??I@)?wak????1?@?϶@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateQ???????!R?ᕪ*@)???o??1?u?$?h@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorҧU??f~?!?6?d?@)ҧU??f~?1?6?d?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapJ?O?c??!??O ?9.@)?߽?Ƅh?1c???y??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?16.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??di????I?3??1@QƤ?DT@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????K???????K???!????K???      ??!       "	??!???#@??!???#@!??!???#@*      ??!       2	S<.?ED??S<.?ED??!S<.?ED??:	7???* @7???* @!7???* @B      ??!       J	O ????O ????!O ????R      ??!       Z	O ????O ????!O ????b      ??!       JGPUY??di????b q?3??1@yƤ?DT@