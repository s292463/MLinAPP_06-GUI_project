	0*?Ш6@0*?Ш6@!0*?Ш6@      ??!       "_
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0*?Ш6@1:??Kt@I?Bs?F?3@r0*	R???yj@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatwg????!??=hC@)????k???1,?Z۷$A@:Preprocessing2U
Iterator::Model::ParallelMapV2dyW=`??!_????0@)dyW=`??1_????0@:Preprocessing2F
Iterator::Model??pvk???!A?HV??>@)΋_?(??1?IHߎ?+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?R@?? ??!k?????(@)?R@?? ??1k?????(@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap
???ç?!ꯉ??5@)>??I????1j?R??"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip3NCT????!??m??XQ@)i???>Ȓ?1Ͽ*[?Q!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory??[Y???!M&??@)y??[Y???1M&??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?87.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIG?????U@Q?m?Y??)@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
      ??!             ??!       "	:??Kt@:??Kt@!:??Kt@*      ??!       2      ??!       :	?Bs?F?3@?Bs?F?3@!?Bs?F?3@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qG?????U@y?m?Y??)@