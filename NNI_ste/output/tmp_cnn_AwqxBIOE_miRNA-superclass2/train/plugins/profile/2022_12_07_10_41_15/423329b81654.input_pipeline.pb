	???:T?@???:T?@!???:T?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC???:T?@]R?????10?AC?@A]?E?~U?I? ??	L @rEagerKernelExecute 0*	㥛? ?f@2F
Iterator::Modela?9???!??? ?F@)??Z&????1Y?*??*?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatQ??lu??!w??ej;@)Z??ڊ???1
b? ?7@:Preprocessing2U
Iterator::Model::ParallelMapV2M????'??!?K?*,@)M????'??1?K?*,@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???ECƓ?!z	B%K%@)???ECƓ?1z	B%K%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?KTol??!mp??_K@)c??Ց??13????%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???[??!MM????.@)a???)??1???8{@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??h??{?!l۽???@)??h??{?1l۽???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?V	?3??!?cc??0@)7ݲC??f?1?S?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?28.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI|6??a?>@Qa2???IQ@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	]R?????]R?????!]R?????      ??!       "	0?AC?@0?AC?@!0?AC?@*      ??!       2	]?E?~U?]?E?~U?!]?E?~U?:	? ??	L @? ??	L @!? ??	L @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q|6??a?>@ya2???IQ@