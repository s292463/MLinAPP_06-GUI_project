	nR?Xۉ?@nR?Xۉ?@!nR?Xۉ?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCnR?Xۉ?@?v??? ??1ŏ1??@A??ڊ?e??I?/???S@rEagerKernelExecute 0*	ףp=
Sb@2F
Iterator::Model????u???!???uշH@)AH0?[??1????@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate??m??!qo?(b?@@)??? 4J??1#?w???@:Preprocessing2U
Iterator::Model::ParallelMapV2L?u?~??!Ҹg?Q2@)L?u?~??1Ҹg?Q2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?4'/2??!V?m?x?$@)(?x?ߢ??1?XT?p)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?D?[????!RTA?*HI@)?????~?1????dU@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??\7??v?!?wK@)??\7??v?1?wK@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapy\T??b??!??Gǟ?A@)?=Զad?1?.?ʳ'??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice?cyW=`^?!M?d?<??)?cyW=`^?1M?d?<??:Preprocessing2?
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensorҏ?S??[?!????V???)ҏ?S??[?1????V???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?12.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIxQ?#??*@Q?u?? ?U@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?v??? ???v??? ??!?v??? ??      ??!       "	ŏ1??@ŏ1??@!ŏ1??@*      ??!       2	??ڊ?e????ڊ?e??!??ڊ?e??:	?/???S@?/???S@!?/???S@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qxQ?#??*@y?u?? ?U@