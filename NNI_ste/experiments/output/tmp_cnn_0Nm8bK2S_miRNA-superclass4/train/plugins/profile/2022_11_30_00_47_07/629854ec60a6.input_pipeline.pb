	???N??3@???N??3@!???N??3@	?:W?h?
@?:W?h?
@!?:W?h?
@"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL???N??3@???)???1???å0@AYİØ???I?-X????Y??<HO??rEagerKernelExecute 0*	???Q?m@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateP?mp???!+rz	3zB@)?3??????1?6??U??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat? %̴??!m1?L8@)??????1w|?sZS5@:Preprocessing2F
Iterator::Model	n?l????!?̙@@)?ʅʿ???1???K3@:Preprocessing2U
Iterator::Model::ParallelMapV2?"?Ƥ??!5VF??)@)?"?Ƥ??15VF??)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??Ry=??!r??A?@)??Ry=??1r??A?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?yUg???!y????P@)??T????1?0??b@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?|A	}?!??4???@)?|A	}?1??4???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???m???!???{?NC@)???Fu:p?17??I????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?8.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?:W?h?
@I?K?d?9(@Q??m?"U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???)??????)???!???)???      ??!       "	???å0@???å0@!???å0@*      ??!       2	YİØ???YİØ???!YİØ???:	?-X?????-X????!?-X????B      ??!       J	??<HO????<HO??!??<HO??R      ??!       Z	??<HO????<HO??!??<HO??b      ??!       JGPUY?:W?h?
@b q?K?d?9(@y??m?"U@