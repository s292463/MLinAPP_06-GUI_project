ѕ
ч
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ЯЭ
~
Conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:%* 
shared_nameConv1D_1/kernel
w
#Conv1D_1/kernel/Read/ReadVariableOpReadVariableOpConv1D_1/kernel*"
_output_shapes
:%*
dtype0
r
Conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:%*
shared_nameConv1D_1/bias
k
!Conv1D_1/bias/Read/ReadVariableOpReadVariableOpConv1D_1/bias*
_output_shapes
:%*
dtype0
~
Conv1D_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:%0* 
shared_nameConv1D_2/kernel
w
#Conv1D_2/kernel/Read/ReadVariableOpReadVariableOpConv1D_2/kernel*"
_output_shapes
:%0*
dtype0
r
Conv1D_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameConv1D_2/bias
k
!Conv1D_2/bias/Read/ReadVariableOpReadVariableOpConv1D_2/bias*
_output_shapes
:0*
dtype0
~
Conv1D_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0`* 
shared_nameConv1D_3/kernel
w
#Conv1D_3/kernel/Read/ReadVariableOpReadVariableOpConv1D_3/kernel*"
_output_shapes
:0`*
dtype0
r
Conv1D_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameConv1D_3/bias
k
!Conv1D_3/bias/Read/ReadVariableOpReadVariableOpConv1D_3/bias*
_output_shapes
:`*
dtype0
~
Conv1D_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:`D* 
shared_nameConv1D_4/kernel
w
#Conv1D_4/kernel/Read/ReadVariableOpReadVariableOpConv1D_4/kernel*"
_output_shapes
:`D*
dtype0
r
Conv1D_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:D*
shared_nameConv1D_4/bias
k
!Conv1D_4/bias/Read/ReadVariableOpReadVariableOpConv1D_4/bias*
_output_shapes
:D*
dtype0
y
Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	S*
shared_nameDense_1/kernel
r
"Dense_1/kernel/Read/ReadVariableOpReadVariableOpDense_1/kernel*
_output_shapes
:	S*
dtype0
p
Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*
shared_nameDense_1/bias
i
 Dense_1/bias/Read/ReadVariableOpReadVariableOpDense_1/bias*
_output_shapes
:S*
dtype0
x
Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:S!*
shared_nameDense_2/kernel
q
"Dense_2/kernel/Read/ReadVariableOpReadVariableOpDense_2/kernel*
_output_shapes

:S!*
dtype0
p
Dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:!*
shared_nameDense_2/bias
i
 Dense_2/bias/Read/ReadVariableOpReadVariableOpDense_2/bias*
_output_shapes
:!*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/Conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:%*'
shared_nameAdam/Conv1D_1/kernel/m

*Adam/Conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/kernel/m*"
_output_shapes
:%*
dtype0

Adam/Conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:%*%
shared_nameAdam/Conv1D_1/bias/m
y
(Adam/Conv1D_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/bias/m*
_output_shapes
:%*
dtype0

Adam/Conv1D_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:%0*'
shared_nameAdam/Conv1D_2/kernel/m

*Adam/Conv1D_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/kernel/m*"
_output_shapes
:%0*
dtype0

Adam/Conv1D_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/Conv1D_2/bias/m
y
(Adam/Conv1D_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/bias/m*
_output_shapes
:0*
dtype0

Adam/Conv1D_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0`*'
shared_nameAdam/Conv1D_3/kernel/m

*Adam/Conv1D_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/kernel/m*"
_output_shapes
:0`*
dtype0

Adam/Conv1D_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/Conv1D_3/bias/m
y
(Adam/Conv1D_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/bias/m*
_output_shapes
:`*
dtype0

Adam/Conv1D_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`D*'
shared_nameAdam/Conv1D_4/kernel/m

*Adam/Conv1D_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/kernel/m*"
_output_shapes
:`D*
dtype0

Adam/Conv1D_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:D*%
shared_nameAdam/Conv1D_4/bias/m
y
(Adam/Conv1D_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/bias/m*
_output_shapes
:D*
dtype0

Adam/Dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	S*&
shared_nameAdam/Dense_1/kernel/m

)Adam/Dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_1/kernel/m*
_output_shapes
:	S*
dtype0
~
Adam/Dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*$
shared_nameAdam/Dense_1/bias/m
w
'Adam/Dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_1/bias/m*
_output_shapes
:S*
dtype0

Adam/Dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:S!*&
shared_nameAdam/Dense_2/kernel/m

)Adam/Dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_2/kernel/m*
_output_shapes

:S!*
dtype0
~
Adam/Dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:!*$
shared_nameAdam/Dense_2/bias/m
w
'Adam/Dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_2/bias/m*
_output_shapes
:!*
dtype0

Adam/Conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:%*'
shared_nameAdam/Conv1D_1/kernel/v

*Adam/Conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/kernel/v*"
_output_shapes
:%*
dtype0

Adam/Conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:%*%
shared_nameAdam/Conv1D_1/bias/v
y
(Adam/Conv1D_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/bias/v*
_output_shapes
:%*
dtype0

Adam/Conv1D_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:%0*'
shared_nameAdam/Conv1D_2/kernel/v

*Adam/Conv1D_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/kernel/v*"
_output_shapes
:%0*
dtype0

Adam/Conv1D_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/Conv1D_2/bias/v
y
(Adam/Conv1D_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/bias/v*
_output_shapes
:0*
dtype0

Adam/Conv1D_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0`*'
shared_nameAdam/Conv1D_3/kernel/v

*Adam/Conv1D_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/kernel/v*"
_output_shapes
:0`*
dtype0

Adam/Conv1D_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/Conv1D_3/bias/v
y
(Adam/Conv1D_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/bias/v*
_output_shapes
:`*
dtype0

Adam/Conv1D_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`D*'
shared_nameAdam/Conv1D_4/kernel/v

*Adam/Conv1D_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/kernel/v*"
_output_shapes
:`D*
dtype0

Adam/Conv1D_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:D*%
shared_nameAdam/Conv1D_4/bias/v
y
(Adam/Conv1D_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/bias/v*
_output_shapes
:D*
dtype0

Adam/Dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	S*&
shared_nameAdam/Dense_1/kernel/v

)Adam/Dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_1/kernel/v*
_output_shapes
:	S*
dtype0
~
Adam/Dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:S*$
shared_nameAdam/Dense_1/bias/v
w
'Adam/Dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_1/bias/v*
_output_shapes
:S*
dtype0

Adam/Dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:S!*&
shared_nameAdam/Dense_2/kernel/v

)Adam/Dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_2/kernel/v*
_output_shapes

:S!*
dtype0
~
Adam/Dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:!*$
shared_nameAdam/Dense_2/bias/v
w
'Adam/Dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_2/bias/v*
_output_shapes
:!*
dtype0

NoOpNoOp
N
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*РM
valueЖMBГM BЌM
Й
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
R
$	variables
%regularization_losses
&trainable_variables
'	keras_api
h

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
R
.	variables
/regularization_losses
0trainable_variables
1	keras_api
h

2kernel
3bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
R
8	variables
9regularization_losses
:trainable_variables
;	keras_api
R
<	variables
=regularization_losses
>trainable_variables
?	keras_api
h

@kernel
Abias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
R
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
h

Jkernel
Kbias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
А
Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_ratemЁmЂmЃmЄ(mЅ)mІ2mЇ3mЈ@mЉAmЊJmЋKmЌv­vЎvЏvА(vБ)vВ2vГ3vД@vЕAvЖJvЗKvИ
V
0
1
2
3
(4
)5
26
37
@8
A9
J10
K11
 
V
0
1
2
3
(4
)5
26
37
@8
A9
J10
K11
­

Ulayers
Vlayer_regularization_losses
	variables
regularization_losses
Wnon_trainable_variables
trainable_variables
Xmetrics
Ylayer_metrics
 
[Y
VARIABLE_VALUEConv1D_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

Zlayers
[layer_regularization_losses
	variables
regularization_losses
\non_trainable_variables
trainable_variables
]metrics
^layer_metrics
 
 
 
­

_layers
`layer_regularization_losses
	variables
regularization_losses
anon_trainable_variables
trainable_variables
bmetrics
clayer_metrics
[Y
VARIABLE_VALUEConv1D_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­

dlayers
elayer_regularization_losses
 	variables
!regularization_losses
fnon_trainable_variables
"trainable_variables
gmetrics
hlayer_metrics
 
 
 
­

ilayers
jlayer_regularization_losses
$	variables
%regularization_losses
knon_trainable_variables
&trainable_variables
lmetrics
mlayer_metrics
[Y
VARIABLE_VALUEConv1D_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
­

nlayers
olayer_regularization_losses
*	variables
+regularization_losses
pnon_trainable_variables
,trainable_variables
qmetrics
rlayer_metrics
 
 
 
­

slayers
tlayer_regularization_losses
.	variables
/regularization_losses
unon_trainable_variables
0trainable_variables
vmetrics
wlayer_metrics
[Y
VARIABLE_VALUEConv1D_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31
 

20
31
­

xlayers
ylayer_regularization_losses
4	variables
5regularization_losses
znon_trainable_variables
6trainable_variables
{metrics
|layer_metrics
 
 
 
Џ

}layers
~layer_regularization_losses
8	variables
9regularization_losses
non_trainable_variables
:trainable_variables
metrics
layer_metrics
 
 
 
В
layers
 layer_regularization_losses
<	variables
=regularization_losses
non_trainable_variables
>trainable_variables
metrics
layer_metrics
ZX
VARIABLE_VALUEDense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1
 

@0
A1
В
layers
 layer_regularization_losses
B	variables
Cregularization_losses
non_trainable_variables
Dtrainable_variables
metrics
layer_metrics
 
 
 
В
layers
 layer_regularization_losses
F	variables
Gregularization_losses
non_trainable_variables
Htrainable_variables
metrics
layer_metrics
ZX
VARIABLE_VALUEDense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
 

J0
K1
В
layers
 layer_regularization_losses
L	variables
Mregularization_losses
non_trainable_variables
Ntrainable_variables
metrics
layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
^
0
1
2
3
4
5
6
7
	8

9
10
11
12
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
 	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
~|
VARIABLE_VALUEAdam/Conv1D_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv1D_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/Conv1D_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Dense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/Dense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*,
_output_shapes
:џџџџџџџџџи*
dtype0*!
shape:џџџџџџџџџи

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Conv1D_1/kernelConv1D_1/biasConv1D_2/kernelConv1D_2/biasConv1D_3/kernelConv1D_3/biasConv1D_4/kernelConv1D_4/biasDense_1/kernelDense_1/biasDense_2/kernelDense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_23525
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#Conv1D_1/kernel/Read/ReadVariableOp!Conv1D_1/bias/Read/ReadVariableOp#Conv1D_2/kernel/Read/ReadVariableOp!Conv1D_2/bias/Read/ReadVariableOp#Conv1D_3/kernel/Read/ReadVariableOp!Conv1D_3/bias/Read/ReadVariableOp#Conv1D_4/kernel/Read/ReadVariableOp!Conv1D_4/bias/Read/ReadVariableOp"Dense_1/kernel/Read/ReadVariableOp Dense_1/bias/Read/ReadVariableOp"Dense_2/kernel/Read/ReadVariableOp Dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/Conv1D_1/kernel/m/Read/ReadVariableOp(Adam/Conv1D_1/bias/m/Read/ReadVariableOp*Adam/Conv1D_2/kernel/m/Read/ReadVariableOp(Adam/Conv1D_2/bias/m/Read/ReadVariableOp*Adam/Conv1D_3/kernel/m/Read/ReadVariableOp(Adam/Conv1D_3/bias/m/Read/ReadVariableOp*Adam/Conv1D_4/kernel/m/Read/ReadVariableOp(Adam/Conv1D_4/bias/m/Read/ReadVariableOp)Adam/Dense_1/kernel/m/Read/ReadVariableOp'Adam/Dense_1/bias/m/Read/ReadVariableOp)Adam/Dense_2/kernel/m/Read/ReadVariableOp'Adam/Dense_2/bias/m/Read/ReadVariableOp*Adam/Conv1D_1/kernel/v/Read/ReadVariableOp(Adam/Conv1D_1/bias/v/Read/ReadVariableOp*Adam/Conv1D_2/kernel/v/Read/ReadVariableOp(Adam/Conv1D_2/bias/v/Read/ReadVariableOp*Adam/Conv1D_3/kernel/v/Read/ReadVariableOp(Adam/Conv1D_3/bias/v/Read/ReadVariableOp*Adam/Conv1D_4/kernel/v/Read/ReadVariableOp(Adam/Conv1D_4/bias/v/Read/ReadVariableOp)Adam/Dense_1/kernel/v/Read/ReadVariableOp'Adam/Dense_1/bias/v/Read/ReadVariableOp)Adam/Dense_2/kernel/v/Read/ReadVariableOp'Adam/Dense_2/bias/v/Read/ReadVariableOpConst*:
Tin3
12/	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_24124
	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv1D_1/kernelConv1D_1/biasConv1D_2/kernelConv1D_2/biasConv1D_3/kernelConv1D_3/biasConv1D_4/kernelConv1D_4/biasDense_1/kernelDense_1/biasDense_2/kernelDense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/Conv1D_1/kernel/mAdam/Conv1D_1/bias/mAdam/Conv1D_2/kernel/mAdam/Conv1D_2/bias/mAdam/Conv1D_3/kernel/mAdam/Conv1D_3/bias/mAdam/Conv1D_4/kernel/mAdam/Conv1D_4/bias/mAdam/Dense_1/kernel/mAdam/Dense_1/bias/mAdam/Dense_2/kernel/mAdam/Dense_2/bias/mAdam/Conv1D_1/kernel/vAdam/Conv1D_1/bias/vAdam/Conv1D_2/kernel/vAdam/Conv1D_2/bias/vAdam/Conv1D_3/kernel/vAdam/Conv1D_3/bias/vAdam/Conv1D_4/kernel/vAdam/Conv1D_4/bias/vAdam/Dense_1/kernel/vAdam/Dense_1/bias/vAdam/Dense_2/kernel/vAdam/Dense_2/bias/v*9
Tin2
02.*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_24269п

p


@__inference_model_layer_call_and_return_conditional_losses_23607

inputsJ
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:%6
(conv1d_1_biasadd_readvariableop_resource:%J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:%06
(conv1d_2_biasadd_readvariableop_resource:0J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:0`6
(conv1d_3_biasadd_readvariableop_resource:`J
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:`D6
(conv1d_4_biasadd_readvariableop_resource:D9
&dense_1_matmul_readvariableop_resource:	S5
'dense_1_biasadd_readvariableop_resource:S8
&dense_2_matmul_readvariableop_resource:S!5
'dense_2_biasadd_readvariableop_resource:!
identityЂConv1D_1/BiasAdd/ReadVariableOpЂ+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpЂConv1D_2/BiasAdd/ReadVariableOpЂ+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpЂConv1D_3/BiasAdd/ReadVariableOpЂ+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOpЂConv1D_4/BiasAdd/ReadVariableOpЂ+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOpЂDense_1/BiasAdd/ReadVariableOpЂDense_1/MatMul/ReadVariableOpЂDense_2/BiasAdd/ReadVariableOpЂDense_2/MatMul/ReadVariableOp
Conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
Conv1D_1/conv1d/ExpandDims/dimВ
Conv1D_1/conv1d/ExpandDims
ExpandDimsinputs'Conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџи2
Conv1D_1/conv1d/ExpandDimsг
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:%*
dtype02-
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp
 Conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_1/conv1d/ExpandDims_1/dimл
Conv1D_1/conv1d/ExpandDims_1
ExpandDims3Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:%2
Conv1D_1/conv1d/ExpandDims_1м
Conv1D_1/conv1dConv2D#Conv1D_1/conv1d/ExpandDims:output:0%Conv1D_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџг%*
paddingVALID*
strides
2
Conv1D_1/conv1dЎ
Conv1D_1/conv1d/SqueezeSqueezeConv1D_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџг%*
squeeze_dims

§џџџџџџџџ2
Conv1D_1/conv1d/SqueezeЇ
Conv1D_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:%*
dtype02!
Conv1D_1/BiasAdd/ReadVariableOpБ
Conv1D_1/BiasAddBiasAdd Conv1D_1/conv1d/Squeeze:output:0'Conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџг%2
Conv1D_1/BiasAddx
Conv1D_1/ReluReluConv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџг%2
Conv1D_1/Relu
MaxPooling1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_1/ExpandDims/dimФ
MaxPooling1D_1/ExpandDims
ExpandDimsConv1D_1/Relu:activations:0&MaxPooling1D_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџг%2
MaxPooling1D_1/ExpandDimsЭ
MaxPooling1D_1/MaxPoolMaxPool"MaxPooling1D_1/ExpandDims:output:0*0
_output_shapes
:џџџџџџџџџ%*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_1/MaxPoolЊ
MaxPooling1D_1/SqueezeSqueezeMaxPooling1D_1/MaxPool:output:0*
T0*,
_output_shapes
:џџџџџџџџџ%*
squeeze_dims
2
MaxPooling1D_1/Squeeze
Conv1D_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
Conv1D_2/conv1d/ExpandDims/dimЫ
Conv1D_2/conv1d/ExpandDims
ExpandDimsMaxPooling1D_1/Squeeze:output:0'Conv1D_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ%2
Conv1D_2/conv1d/ExpandDimsг
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:%0*
dtype02-
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp
 Conv1D_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_2/conv1d/ExpandDims_1/dimл
Conv1D_2/conv1d/ExpandDims_1
ExpandDims3Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:%02
Conv1D_2/conv1d/ExpandDims_1м
Conv1D_2/conv1dConv2D#Conv1D_2/conv1d/ExpandDims:output:0%Conv1D_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ0*
paddingVALID*
strides
2
Conv1D_2/conv1dЎ
Conv1D_2/conv1d/SqueezeSqueezeConv1D_2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ0*
squeeze_dims

§џџџџџџџџ2
Conv1D_2/conv1d/SqueezeЇ
Conv1D_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
Conv1D_2/BiasAdd/ReadVariableOpБ
Conv1D_2/BiasAddBiasAdd Conv1D_2/conv1d/Squeeze:output:0'Conv1D_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ02
Conv1D_2/BiasAddx
Conv1D_2/ReluReluConv1D_2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ02
Conv1D_2/Relu
MaxPooling1D_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_2/ExpandDims/dimФ
MaxPooling1D_2/ExpandDims
ExpandDimsConv1D_2/Relu:activations:0&MaxPooling1D_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ02
MaxPooling1D_2/ExpandDimsЬ
MaxPooling1D_2/MaxPoolMaxPool"MaxPooling1D_2/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ$0*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_2/MaxPoolЉ
MaxPooling1D_2/SqueezeSqueezeMaxPooling1D_2/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$0*
squeeze_dims
2
MaxPooling1D_2/Squeeze
Conv1D_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
Conv1D_3/conv1d/ExpandDims/dimЪ
Conv1D_3/conv1d/ExpandDims
ExpandDimsMaxPooling1D_2/Squeeze:output:0'Conv1D_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$02
Conv1D_3/conv1d/ExpandDimsг
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0`*
dtype02-
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp
 Conv1D_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_3/conv1d/ExpandDims_1/dimл
Conv1D_3/conv1d/ExpandDims_1
ExpandDims3Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:0`2
Conv1D_3/conv1d/ExpandDims_1л
Conv1D_3/conv1dConv2D#Conv1D_3/conv1d/ExpandDims:output:0%Conv1D_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ!`*
paddingVALID*
strides
2
Conv1D_3/conv1d­
Conv1D_3/conv1d/SqueezeSqueezeConv1D_3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ!`*
squeeze_dims

§џџџџџџџџ2
Conv1D_3/conv1d/SqueezeЇ
Conv1D_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02!
Conv1D_3/BiasAdd/ReadVariableOpА
Conv1D_3/BiasAddBiasAdd Conv1D_3/conv1d/Squeeze:output:0'Conv1D_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ!`2
Conv1D_3/BiasAddw
Conv1D_3/ReluReluConv1D_3/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ!`2
Conv1D_3/Relu
MaxPooling1D_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_3/ExpandDims/dimУ
MaxPooling1D_3/ExpandDims
ExpandDimsConv1D_3/Relu:activations:0&MaxPooling1D_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ!`2
MaxPooling1D_3/ExpandDimsЬ
MaxPooling1D_3/MaxPoolMaxPool"MaxPooling1D_3/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ`*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_3/MaxPoolЉ
MaxPooling1D_3/SqueezeSqueezeMaxPooling1D_3/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ`*
squeeze_dims
2
MaxPooling1D_3/Squeeze
Conv1D_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
Conv1D_4/conv1d/ExpandDims/dimЪ
Conv1D_4/conv1d/ExpandDims
ExpandDimsMaxPooling1D_3/Squeeze:output:0'Conv1D_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`2
Conv1D_4/conv1d/ExpandDimsг
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:`D*
dtype02-
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp
 Conv1D_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_4/conv1d/ExpandDims_1/dimл
Conv1D_4/conv1d/ExpandDims_1
ExpandDims3Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:`D2
Conv1D_4/conv1d/ExpandDims_1л
Conv1D_4/conv1dConv2D#Conv1D_4/conv1d/ExpandDims:output:0%Conv1D_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџD*
paddingVALID*
strides
2
Conv1D_4/conv1d­
Conv1D_4/conv1d/SqueezeSqueezeConv1D_4/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџD*
squeeze_dims

§џџџџџџџџ2
Conv1D_4/conv1d/SqueezeЇ
Conv1D_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:D*
dtype02!
Conv1D_4/BiasAdd/ReadVariableOpА
Conv1D_4/BiasAddBiasAdd Conv1D_4/conv1d/Squeeze:output:0'Conv1D_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџD2
Conv1D_4/BiasAddw
Conv1D_4/ReluReluConv1D_4/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџD2
Conv1D_4/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
flatten/Const
flatten/ReshapeReshapeConv1D_4/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten/Reshape}
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/IdentityІ
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	S*
dtype02
Dense_1/MatMul/ReadVariableOp
Dense_1/MatMulMatMuldropout/Identity:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџS2
Dense_1/MatMulЄ
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:S*
dtype02 
Dense_1/BiasAdd/ReadVariableOpЁ
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџS2
Dense_1/BiasAddp
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџS2
Dense_1/Relu
dropout_1/IdentityIdentityDense_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџS2
dropout_1/IdentityЅ
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:S!*
dtype02
Dense_2/MatMul/ReadVariableOp 
Dense_2/MatMulMatMuldropout_1/Identity:output:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ!2
Dense_2/MatMulЄ
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype02 
Dense_2/BiasAdd/ReadVariableOpЁ
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ!2
Dense_2/BiasAddy
Dense_2/SoftmaxSoftmaxDense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ!2
Dense_2/SoftmaxЏ
IdentityIdentityDense_2/Softmax:softmax:0 ^Conv1D_1/BiasAdd/ReadVariableOp,^Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_2/BiasAdd/ReadVariableOp,^Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_3/BiasAdd/ReadVariableOp,^Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_4/BiasAdd/ReadVariableOp,^Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџи: : : : : : : : : : : : 2B
Conv1D_1/BiasAdd/ReadVariableOpConv1D_1/BiasAdd/ReadVariableOp2Z
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_2/BiasAdd/ReadVariableOpConv1D_2/BiasAdd/ReadVariableOp2Z
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_3/BiasAdd/ReadVariableOpConv1D_3/BiasAdd/ReadVariableOp2Z
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_4/BiasAdd/ReadVariableOpConv1D_4/BiasAdd/ReadVariableOp2Z
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2@
Dense_2/BiasAdd/ReadVariableOpDense_2/BiasAdd/ReadVariableOp2>
Dense_2/MatMul/ReadVariableOpDense_2/MatMul/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџи
 
_user_specified_nameinputs
з

В
%__inference_model_layer_call_fn_23732

inputs
unknown:%
	unknown_0:%
	unknown_1:%0
	unknown_2:0
	unknown_3:0`
	unknown_4:`
	unknown_5:`D
	unknown_6:D
	unknown_7:	S
	unknown_8:S
	unknown_9:S!

unknown_10:!
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_231422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџи: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџи
 
_user_specified_nameinputs
П
E
)__inference_dropout_1_layer_call_fn_23941

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџS* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_231222
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџS:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs


'__inference_Dense_1_layer_call_fn_23919

inputs
unknown:	S
	unknown_0:S
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџS*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_231112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

e
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_22966

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

ExpandDimsБ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
В

ѓ
B__inference_Dense_2_layer_call_and_return_conditional_losses_23135

inputs0
matmul_readvariableop_resource:S!-
biasadd_readvariableop_resource:!
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:S!*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ!2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:!*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ!2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ!2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџS: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs
м

C__inference_Conv1D_2_layer_call_and_return_conditional_losses_23033

inputsA
+conv1d_expanddims_1_readvariableop_resource:%0-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ%2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:%0*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:%02
conv1d/ExpandDims_1И
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ0*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ0*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ02	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ02
ReluЉ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:џџџџџџџџџ02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ%
 
_user_specified_nameinputs
з6

@__inference_model_layer_call_and_return_conditional_losses_23352

inputs$
conv1d_1_23315:%
conv1d_1_23317:%$
conv1d_2_23321:%0
conv1d_2_23323:0$
conv1d_3_23327:0`
conv1d_3_23329:`$
conv1d_4_23333:`D
conv1d_4_23335:D 
dense_1_23340:	S
dense_1_23342:S
dense_2_23346:S!
dense_2_23348:!
identityЂ Conv1D_1/StatefulPartitionedCallЂ Conv1D_2/StatefulPartitionedCallЂ Conv1D_3/StatefulPartitionedCallЂ Conv1D_4/StatefulPartitionedCallЂDense_1/StatefulPartitionedCallЂDense_2/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCall
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1_23315conv1d_1_23317*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџг%*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_230102"
 Conv1D_1/StatefulPartitionedCall
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ%* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_229512 
MaxPooling1D_1/PartitionedCallК
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0conv1d_2_23321conv1d_2_23323*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_230332"
 Conv1D_2/StatefulPartitionedCall
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_229662 
MaxPooling1D_2/PartitionedCallЙ
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0conv1d_3_23327conv1d_3_23329*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ!`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_230562"
 Conv1D_3/StatefulPartitionedCall
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_229812 
MaxPooling1D_3/PartitionedCallЙ
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0conv1d_4_23333conv1d_4_23335*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџD*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_230792"
 Conv1D_4/StatefulPartitionedCallї
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_230912
flatten/PartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_232322!
dropout/StatefulPartitionedCallБ
Dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_23340dense_1_23342*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџS*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_231112!
Dense_1/StatefulPartitionedCallЕ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџS* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_231992#
!dropout_1/StatefulPartitionedCallГ
Dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_23346dense_2_23348*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_231352!
Dense_2/StatefulPartitionedCall
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџи: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџи
 
_user_specified_nameinputs
Ў

є
B__inference_Dense_1_layer_call_and_return_conditional_losses_23910

inputs1
matmul_readvariableop_resource:	S-
biasadd_readvariableop_resource:S
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	S*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџS2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:S*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџS2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџS2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Т
a
B__inference_dropout_layer_call_and_return_conditional_losses_23889

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЁWЉ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЦ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0*
seed2џџџџ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fўy>2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_23199

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *&дс?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџS2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeХ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџS*
dtype0*
seed2џџџџ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *8Ьн>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџS2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџS2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџS2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџS:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs
Ы
`
'__inference_dropout_layer_call_fn_23899

inputs
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_232322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
д

C__inference_Conv1D_3_layer_call_and_return_conditional_losses_23056

inputsA
+conv1d_expanddims_1_readvariableop_resource:0`-
biasadd_readvariableop_resource:`
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$02
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0`*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:0`2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ!`*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ!`*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ!`2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ!`2
ReluЈ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ!`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ$0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ$0
 
_user_specified_nameinputs
к6

@__inference_model_layer_call_and_return_conditional_losses_23488
input_1$
conv1d_1_23451:%
conv1d_1_23453:%$
conv1d_2_23457:%0
conv1d_2_23459:0$
conv1d_3_23463:0`
conv1d_3_23465:`$
conv1d_4_23469:`D
conv1d_4_23471:D 
dense_1_23476:	S
dense_1_23478:S
dense_2_23482:S!
dense_2_23484:!
identityЂ Conv1D_1/StatefulPartitionedCallЂ Conv1D_2/StatefulPartitionedCallЂ Conv1D_3/StatefulPartitionedCallЂ Conv1D_4/StatefulPartitionedCallЂDense_1/StatefulPartitionedCallЂDense_2/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ!dropout_1/StatefulPartitionedCall
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1_23451conv1d_1_23453*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџг%*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_230102"
 Conv1D_1/StatefulPartitionedCall
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ%* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_229512 
MaxPooling1D_1/PartitionedCallК
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0conv1d_2_23457conv1d_2_23459*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_230332"
 Conv1D_2/StatefulPartitionedCall
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_229662 
MaxPooling1D_2/PartitionedCallЙ
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0conv1d_3_23463conv1d_3_23465*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ!`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_230562"
 Conv1D_3/StatefulPartitionedCall
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_229812 
MaxPooling1D_3/PartitionedCallЙ
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0conv1d_4_23469conv1d_4_23471*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџD*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_230792"
 Conv1D_4/StatefulPartitionedCallї
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_230912
flatten/PartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_232322!
dropout/StatefulPartitionedCallБ
Dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_23476dense_1_23478*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџS*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_231112!
Dense_1/StatefulPartitionedCallЕ
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџS* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_231992#
!dropout_1/StatefulPartitionedCallГ
Dense_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_2_23482dense_2_23484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_231352!
Dense_2/StatefulPartitionedCall
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџи: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall:U Q
,
_output_shapes
:џџџџџџџџџи
!
_user_specified_name	input_1
Ў

є
B__inference_Dense_1_layer_call_and_return_conditional_losses_23111

inputs1
matmul_readvariableop_resource:	S-
biasadd_readvariableop_resource:S
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	S*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџS2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:S*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџS2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџS2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
м

C__inference_Conv1D_1_layer_call_and_return_conditional_losses_23010

inputsA
+conv1d_expanddims_1_readvariableop_resource:%-
biasadd_readvariableop_resource:%
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџи2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:%*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:%2
conv1d/ExpandDims_1И
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџг%*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџг%*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:%*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџг%2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџг%2
ReluЉ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:џџџџџџџџџг%2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџи: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџи
 
_user_specified_nameinputs
р3
Н
@__inference_model_layer_call_and_return_conditional_losses_23448
input_1$
conv1d_1_23411:%
conv1d_1_23413:%$
conv1d_2_23417:%0
conv1d_2_23419:0$
conv1d_3_23423:0`
conv1d_3_23425:`$
conv1d_4_23429:`D
conv1d_4_23431:D 
dense_1_23436:	S
dense_1_23438:S
dense_2_23442:S!
dense_2_23444:!
identityЂ Conv1D_1/StatefulPartitionedCallЂ Conv1D_2/StatefulPartitionedCallЂ Conv1D_3/StatefulPartitionedCallЂ Conv1D_4/StatefulPartitionedCallЂDense_1/StatefulPartitionedCallЂDense_2/StatefulPartitionedCall
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1_23411conv1d_1_23413*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџг%*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_230102"
 Conv1D_1/StatefulPartitionedCall
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ%* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_229512 
MaxPooling1D_1/PartitionedCallК
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0conv1d_2_23417conv1d_2_23419*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_230332"
 Conv1D_2/StatefulPartitionedCall
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_229662 
MaxPooling1D_2/PartitionedCallЙ
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0conv1d_3_23423conv1d_3_23425*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ!`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_230562"
 Conv1D_3/StatefulPartitionedCall
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_229812 
MaxPooling1D_3/PartitionedCallЙ
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0conv1d_4_23429conv1d_4_23431*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџD*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_230792"
 Conv1D_4/StatefulPartitionedCallї
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_230912
flatten/PartitionedCallю
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_230982
dropout/PartitionedCallЉ
Dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_23436dense_1_23438*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџS*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_231112!
Dense_1/StatefulPartitionedCallћ
dropout_1/PartitionedCallPartitionedCall(Dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџS* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_231222
dropout_1/PartitionedCallЋ
Dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_23442dense_2_23444*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_231352!
Dense_2/StatefulPartitionedCallЬ
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџи: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall:U Q
,
_output_shapes
:џџџџџџџџџи
!
_user_specified_name	input_1

e
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_22981

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

ExpandDimsБ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
м

C__inference_Conv1D_1_layer_call_and_return_conditional_losses_23777

inputsA
+conv1d_expanddims_1_readvariableop_resource:%-
biasadd_readvariableop_resource:%
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџи2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:%*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:%2
conv1d/ExpandDims_1И
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџг%*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџг%*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:%*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџг%2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџг%2
ReluЉ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:џџџџџџџџџг%2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџи: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџи
 
_user_specified_nameinputs
Х
C
'__inference_flatten_layer_call_fn_23872

inputs
identityФ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_230912
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџD:S O
+
_output_shapes
:џџџџџџџџџD
 
_user_specified_nameinputs
ЭР
 
!__inference__traced_restore_24269
file_prefix6
 assignvariableop_conv1d_1_kernel:%.
 assignvariableop_1_conv1d_1_bias:%8
"assignvariableop_2_conv1d_2_kernel:%0.
 assignvariableop_3_conv1d_2_bias:08
"assignvariableop_4_conv1d_3_kernel:0`.
 assignvariableop_5_conv1d_3_bias:`8
"assignvariableop_6_conv1d_4_kernel:`D.
 assignvariableop_7_conv1d_4_bias:D4
!assignvariableop_8_dense_1_kernel:	S-
assignvariableop_9_dense_1_bias:S4
"assignvariableop_10_dense_2_kernel:S!.
 assignvariableop_11_dense_2_bias:!'
assignvariableop_12_adam_iter:	 )
assignvariableop_13_adam_beta_1: )
assignvariableop_14_adam_beta_2: (
assignvariableop_15_adam_decay: 0
&assignvariableop_16_adam_learning_rate: #
assignvariableop_17_total: #
assignvariableop_18_count: %
assignvariableop_19_total_1: %
assignvariableop_20_count_1: @
*assignvariableop_21_adam_conv1d_1_kernel_m:%6
(assignvariableop_22_adam_conv1d_1_bias_m:%@
*assignvariableop_23_adam_conv1d_2_kernel_m:%06
(assignvariableop_24_adam_conv1d_2_bias_m:0@
*assignvariableop_25_adam_conv1d_3_kernel_m:0`6
(assignvariableop_26_adam_conv1d_3_bias_m:`@
*assignvariableop_27_adam_conv1d_4_kernel_m:`D6
(assignvariableop_28_adam_conv1d_4_bias_m:D<
)assignvariableop_29_adam_dense_1_kernel_m:	S5
'assignvariableop_30_adam_dense_1_bias_m:S;
)assignvariableop_31_adam_dense_2_kernel_m:S!5
'assignvariableop_32_adam_dense_2_bias_m:!@
*assignvariableop_33_adam_conv1d_1_kernel_v:%6
(assignvariableop_34_adam_conv1d_1_bias_v:%@
*assignvariableop_35_adam_conv1d_2_kernel_v:%06
(assignvariableop_36_adam_conv1d_2_bias_v:0@
*assignvariableop_37_adam_conv1d_3_kernel_v:0`6
(assignvariableop_38_adam_conv1d_3_bias_v:`@
*assignvariableop_39_adam_conv1d_4_kernel_v:`D6
(assignvariableop_40_adam_conv1d_4_bias_v:D<
)assignvariableop_41_adam_dense_1_kernel_v:	S5
'assignvariableop_42_adam_dense_1_bias_v:S;
)assignvariableop_43_adam_dense_2_kernel_v:S!5
'assignvariableop_44_adam_dense_2_bias_v:!
identity_46ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Р
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ь
valueТBП.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesъ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ю
_output_shapesЛ
И::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv1d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ѕ
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ї
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv1d_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѕ
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv1d_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ї
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv1d_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѕ
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv1d_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8І
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Є
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Њ
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ј
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12Ѕ
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ї
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ї
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15І
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ў
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ё
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ё
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ѓ
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ѓ
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21В
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv1d_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22А
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv1d_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23В
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv1d_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24А
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv1d_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25В
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv1d_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26А
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv1d_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27В
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv1d_4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28А
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv1d_4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Б
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Џ
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Б
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Џ
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33В
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv1d_1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34А
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv1d_1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35В
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv1d_2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36А
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv1d_2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37В
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv1d_3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38А
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv1d_3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39В
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv1d_4_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40А
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv1d_4_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Б
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Џ
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Б
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Џ
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpМ
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45Џ
Identity_46IdentityIdentity_45:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_46"#
identity_46Identity_46:output:0*o
_input_shapes^
\: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ё
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_23122

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџS2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџS2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџS:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs


'__inference_Dense_2_layer_call_fn_23966

inputs
unknown:S!
	unknown_0:!
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_231352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџS: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs
Ђ
J
.__inference_MaxPooling1D_2_layer_call_fn_22972

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_229662
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Д

(__inference_Conv1D_1_layer_call_fn_23786

inputs
unknown:%
	unknown_0:%
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџг%*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_230102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџг%2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџи: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџи
 
_user_specified_nameinputs
Ђ
J
.__inference_MaxPooling1D_3_layer_call_fn_22987

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_229812
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
н3
М
@__inference_model_layer_call_and_return_conditional_losses_23142

inputs$
conv1d_1_23011:%
conv1d_1_23013:%$
conv1d_2_23034:%0
conv1d_2_23036:0$
conv1d_3_23057:0`
conv1d_3_23059:`$
conv1d_4_23080:`D
conv1d_4_23082:D 
dense_1_23112:	S
dense_1_23114:S
dense_2_23136:S!
dense_2_23138:!
identityЂ Conv1D_1/StatefulPartitionedCallЂ Conv1D_2/StatefulPartitionedCallЂ Conv1D_3/StatefulPartitionedCallЂ Conv1D_4/StatefulPartitionedCallЂDense_1/StatefulPartitionedCallЂDense_2/StatefulPartitionedCall
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1_23011conv1d_1_23013*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџг%*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_230102"
 Conv1D_1/StatefulPartitionedCall
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ%* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_229512 
MaxPooling1D_1/PartitionedCallК
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0conv1d_2_23034conv1d_2_23036*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_230332"
 Conv1D_2/StatefulPartitionedCall
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ$0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_229662 
MaxPooling1D_2/PartitionedCallЙ
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0conv1d_3_23057conv1d_3_23059*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ!`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_230562"
 Conv1D_3/StatefulPartitionedCall
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ`* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_229812 
MaxPooling1D_3/PartitionedCallЙ
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0conv1d_4_23080conv1d_4_23082*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџD*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_230792"
 Conv1D_4/StatefulPartitionedCallї
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_230912
flatten/PartitionedCallю
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_230982
dropout/PartitionedCallЉ
Dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_23112dense_1_23114*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџS*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_231112!
Dense_1/StatefulPartitionedCallћ
dropout_1/PartitionedCallPartitionedCall(Dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџS* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_231222
dropout_1/PartitionedCallЋ
Dense_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_2_23136dense_2_23138*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_231352!
Dense_2/StatefulPartitionedCallЬ
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџи: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџи
 
_user_specified_nameinputs
м
^
B__inference_flatten_layer_call_and_return_conditional_losses_23091

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџD:S O
+
_output_shapes
:џџџџџџџџџD
 
_user_specified_nameinputs
х\

__inference__traced_save_24124
file_prefix.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop.
*savev2_conv1d_3_kernel_read_readvariableop,
(savev2_conv1d_3_bias_read_readvariableop.
*savev2_conv1d_4_kernel_read_readvariableop,
(savev2_conv1d_4_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableop5
1savev2_adam_conv1d_2_kernel_m_read_readvariableop3
/savev2_adam_conv1d_2_bias_m_read_readvariableop5
1savev2_adam_conv1d_3_kernel_m_read_readvariableop3
/savev2_adam_conv1d_3_bias_m_read_readvariableop5
1savev2_adam_conv1d_4_kernel_m_read_readvariableop3
/savev2_adam_conv1d_4_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableop5
1savev2_adam_conv1d_2_kernel_v_read_readvariableop3
/savev2_adam_conv1d_2_bias_v_read_readvariableop5
1savev2_adam_conv1d_3_kernel_v_read_readvariableop3
/savev2_adam_conv1d_3_bias_v_read_readvariableop5
1savev2_adam_conv1d_4_kernel_v_read_readvariableop3
/savev2_adam_conv1d_4_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameК
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*Ь
valueТBП.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesф
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesч
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop*savev2_conv1d_4_kernel_read_readvariableop(savev2_conv1d_4_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop1savev2_adam_conv1d_2_kernel_m_read_readvariableop/savev2_adam_conv1d_2_bias_m_read_readvariableop1savev2_adam_conv1d_3_kernel_m_read_readvariableop/savev2_adam_conv1d_3_bias_m_read_readvariableop1savev2_adam_conv1d_4_kernel_m_read_readvariableop/savev2_adam_conv1d_4_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop1savev2_adam_conv1d_2_kernel_v_read_readvariableop/savev2_adam_conv1d_2_bias_v_read_readvariableop1savev2_adam_conv1d_3_kernel_v_read_readvariableop/savev2_adam_conv1d_3_bias_v_read_readvariableop1savev2_adam_conv1d_4_kernel_v_read_readvariableop/savev2_adam_conv1d_4_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ў
_input_shapesь
щ: :%:%:%0:0:0`:`:`D:D:	S:S:S!:!: : : : : : : : : :%:%:%0:0:0`:`:`D:D:	S:S:S!:!:%:%:%0:0:0`:`:`D:D:	S:S:S!:!: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:%: 

_output_shapes
:%:($
"
_output_shapes
:%0: 

_output_shapes
:0:($
"
_output_shapes
:0`: 

_output_shapes
:`:($
"
_output_shapes
:`D: 

_output_shapes
:D:%	!

_output_shapes
:	S: 


_output_shapes
:S:$ 

_output_shapes

:S!: 

_output_shapes
:!:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :($
"
_output_shapes
:%: 

_output_shapes
:%:($
"
_output_shapes
:%0: 

_output_shapes
:0:($
"
_output_shapes
:0`: 

_output_shapes
:`:($
"
_output_shapes
:`D: 

_output_shapes
:D:%!

_output_shapes
:	S: 

_output_shapes
:S:$  

_output_shapes

:S!: !

_output_shapes
:!:("$
"
_output_shapes
:%: #

_output_shapes
:%:($$
"
_output_shapes
:%0: %

_output_shapes
:0:(&$
"
_output_shapes
:0`: '

_output_shapes
:`:(($
"
_output_shapes
:`D: )

_output_shapes
:D:%*!

_output_shapes
:	S: +

_output_shapes
:S:$, 

_output_shapes

:S!: -

_output_shapes
:!:.

_output_shapes
: 
ё
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_23924

inputs

identity_1Z
IdentityIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџS2

Identityi

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:џџџџџџџџџS2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџS:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs
Ы
b
)__inference_dropout_1_layer_call_fn_23946

inputs
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџS* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_231992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџS22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs
Т
a
B__inference_dropout_layer_call_and_return_conditional_losses_23232

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЁWЉ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЦ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0*
seed2џџџџ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fўy>2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ђ
J
.__inference_MaxPooling1D_1_layer_call_fn_22957

inputs
identityр
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_229512
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
д

C__inference_Conv1D_4_layer_call_and_return_conditional_losses_23079

inputsA
+conv1d_expanddims_1_readvariableop_resource:`D-
biasadd_readvariableop_resource:D
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:`D*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:`D2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџD*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџD*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:D*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџD2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџD2
ReluЈ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџD2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ`
 
_user_specified_nameinputs
Л
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_23936

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *&дс?2
dropout/Consts
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџS2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeХ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџS*
dtype0*
seed2џџџџ2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *8Ьн>2
dropout/GreaterEqual/yО
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџS2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџS2
dropout/Castz
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџS2
dropout/Mul_1e
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџS:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs
м
^
B__inference_flatten_layer_call_and_return_conditional_losses_23867

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџD:S O
+
_output_shapes
:џџџџџџџџџD
 
_user_specified_nameinputs
к

Г
%__inference_model_layer_call_fn_23408
input_1
unknown:%
	unknown_0:%
	unknown_1:%0
	unknown_2:0
	unknown_3:0`
	unknown_4:`
	unknown_5:`D
	unknown_6:D
	unknown_7:	S
	unknown_8:S
	unknown_9:S!

unknown_10:!
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_233522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџи: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:џџџџџџџџџи
!
_user_specified_name	input_1
ѓ
`
B__inference_dropout_layer_call_and_return_conditional_losses_23098

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Д

(__inference_Conv1D_2_layer_call_fn_23811

inputs
unknown:%0
	unknown_0:0
identityЂStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_230332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ%: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ%
 
_user_specified_nameinputs
д

C__inference_Conv1D_3_layer_call_and_return_conditional_losses_23827

inputsA
+conv1d_expanddims_1_readvariableop_resource:0`-
biasadd_readvariableop_resource:`
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$02
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0`*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:0`2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ!`*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ!`*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ!`2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ!`2
ReluЈ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџ!`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ$0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ$0
 
_user_specified_nameinputs
П
C
'__inference_dropout_layer_call_fn_23894

inputs
identityФ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_230982
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

e
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_22951

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

ExpandDimsБ
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Х|

 __inference__wrapped_model_22942
input_1P
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource:%<
.model_conv1d_1_biasadd_readvariableop_resource:%P
:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource:%0<
.model_conv1d_2_biasadd_readvariableop_resource:0P
:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource:0`<
.model_conv1d_3_biasadd_readvariableop_resource:`P
:model_conv1d_4_conv1d_expanddims_1_readvariableop_resource:`D<
.model_conv1d_4_biasadd_readvariableop_resource:D?
,model_dense_1_matmul_readvariableop_resource:	S;
-model_dense_1_biasadd_readvariableop_resource:S>
,model_dense_2_matmul_readvariableop_resource:S!;
-model_dense_2_biasadd_readvariableop_resource:!
identityЂ%model/Conv1D_1/BiasAdd/ReadVariableOpЂ1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpЂ%model/Conv1D_2/BiasAdd/ReadVariableOpЂ1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpЂ%model/Conv1D_3/BiasAdd/ReadVariableOpЂ1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOpЂ%model/Conv1D_4/BiasAdd/ReadVariableOpЂ1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOpЂ$model/Dense_1/BiasAdd/ReadVariableOpЂ#model/Dense_1/MatMul/ReadVariableOpЂ$model/Dense_2/BiasAdd/ReadVariableOpЂ#model/Dense_2/MatMul/ReadVariableOp
$model/Conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2&
$model/Conv1D_1/conv1d/ExpandDims/dimХ
 model/Conv1D_1/conv1d/ExpandDims
ExpandDimsinput_1-model/Conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџи2"
 model/Conv1D_1/conv1d/ExpandDimsх
1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:%*
dtype023
1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp
&model/Conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_1/conv1d/ExpandDims_1/dimѓ
"model/Conv1D_1/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:%2$
"model/Conv1D_1/conv1d/ExpandDims_1є
model/Conv1D_1/conv1dConv2D)model/Conv1D_1/conv1d/ExpandDims:output:0+model/Conv1D_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџг%*
paddingVALID*
strides
2
model/Conv1D_1/conv1dР
model/Conv1D_1/conv1d/SqueezeSqueezemodel/Conv1D_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџг%*
squeeze_dims

§џџџџџџџџ2
model/Conv1D_1/conv1d/SqueezeЙ
%model/Conv1D_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:%*
dtype02'
%model/Conv1D_1/BiasAdd/ReadVariableOpЩ
model/Conv1D_1/BiasAddBiasAdd&model/Conv1D_1/conv1d/Squeeze:output:0-model/Conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџг%2
model/Conv1D_1/BiasAdd
model/Conv1D_1/ReluRelumodel/Conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџг%2
model/Conv1D_1/Relu
#model/MaxPooling1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/MaxPooling1D_1/ExpandDims/dimм
model/MaxPooling1D_1/ExpandDims
ExpandDims!model/Conv1D_1/Relu:activations:0,model/MaxPooling1D_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџг%2!
model/MaxPooling1D_1/ExpandDimsп
model/MaxPooling1D_1/MaxPoolMaxPool(model/MaxPooling1D_1/ExpandDims:output:0*0
_output_shapes
:џџџџџџџџџ%*
ksize
*
paddingVALID*
strides
2
model/MaxPooling1D_1/MaxPoolМ
model/MaxPooling1D_1/SqueezeSqueeze%model/MaxPooling1D_1/MaxPool:output:0*
T0*,
_output_shapes
:џџџџџџџџџ%*
squeeze_dims
2
model/MaxPooling1D_1/Squeeze
$model/Conv1D_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2&
$model/Conv1D_2/conv1d/ExpandDims/dimу
 model/Conv1D_2/conv1d/ExpandDims
ExpandDims%model/MaxPooling1D_1/Squeeze:output:0-model/Conv1D_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ%2"
 model/Conv1D_2/conv1d/ExpandDimsх
1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:%0*
dtype023
1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp
&model/Conv1D_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_2/conv1d/ExpandDims_1/dimѓ
"model/Conv1D_2/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:%02$
"model/Conv1D_2/conv1d/ExpandDims_1є
model/Conv1D_2/conv1dConv2D)model/Conv1D_2/conv1d/ExpandDims:output:0+model/Conv1D_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ0*
paddingVALID*
strides
2
model/Conv1D_2/conv1dР
model/Conv1D_2/conv1d/SqueezeSqueezemodel/Conv1D_2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ0*
squeeze_dims

§џџџџџџџџ2
model/Conv1D_2/conv1d/SqueezeЙ
%model/Conv1D_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02'
%model/Conv1D_2/BiasAdd/ReadVariableOpЩ
model/Conv1D_2/BiasAddBiasAdd&model/Conv1D_2/conv1d/Squeeze:output:0-model/Conv1D_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ02
model/Conv1D_2/BiasAdd
model/Conv1D_2/ReluRelumodel/Conv1D_2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ02
model/Conv1D_2/Relu
#model/MaxPooling1D_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/MaxPooling1D_2/ExpandDims/dimм
model/MaxPooling1D_2/ExpandDims
ExpandDims!model/Conv1D_2/Relu:activations:0,model/MaxPooling1D_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ02!
model/MaxPooling1D_2/ExpandDimsо
model/MaxPooling1D_2/MaxPoolMaxPool(model/MaxPooling1D_2/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ$0*
ksize
*
paddingVALID*
strides
2
model/MaxPooling1D_2/MaxPoolЛ
model/MaxPooling1D_2/SqueezeSqueeze%model/MaxPooling1D_2/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$0*
squeeze_dims
2
model/MaxPooling1D_2/Squeeze
$model/Conv1D_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2&
$model/Conv1D_3/conv1d/ExpandDims/dimт
 model/Conv1D_3/conv1d/ExpandDims
ExpandDims%model/MaxPooling1D_2/Squeeze:output:0-model/Conv1D_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$02"
 model/Conv1D_3/conv1d/ExpandDimsх
1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0`*
dtype023
1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp
&model/Conv1D_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_3/conv1d/ExpandDims_1/dimѓ
"model/Conv1D_3/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:0`2$
"model/Conv1D_3/conv1d/ExpandDims_1ѓ
model/Conv1D_3/conv1dConv2D)model/Conv1D_3/conv1d/ExpandDims:output:0+model/Conv1D_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ!`*
paddingVALID*
strides
2
model/Conv1D_3/conv1dП
model/Conv1D_3/conv1d/SqueezeSqueezemodel/Conv1D_3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ!`*
squeeze_dims

§џџџџџџџџ2
model/Conv1D_3/conv1d/SqueezeЙ
%model/Conv1D_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02'
%model/Conv1D_3/BiasAdd/ReadVariableOpШ
model/Conv1D_3/BiasAddBiasAdd&model/Conv1D_3/conv1d/Squeeze:output:0-model/Conv1D_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ!`2
model/Conv1D_3/BiasAdd
model/Conv1D_3/ReluRelumodel/Conv1D_3/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ!`2
model/Conv1D_3/Relu
#model/MaxPooling1D_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/MaxPooling1D_3/ExpandDims/dimл
model/MaxPooling1D_3/ExpandDims
ExpandDims!model/Conv1D_3/Relu:activations:0,model/MaxPooling1D_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ!`2!
model/MaxPooling1D_3/ExpandDimsо
model/MaxPooling1D_3/MaxPoolMaxPool(model/MaxPooling1D_3/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ`*
ksize
*
paddingVALID*
strides
2
model/MaxPooling1D_3/MaxPoolЛ
model/MaxPooling1D_3/SqueezeSqueeze%model/MaxPooling1D_3/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ`*
squeeze_dims
2
model/MaxPooling1D_3/Squeeze
$model/Conv1D_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2&
$model/Conv1D_4/conv1d/ExpandDims/dimт
 model/Conv1D_4/conv1d/ExpandDims
ExpandDims%model/MaxPooling1D_3/Squeeze:output:0-model/Conv1D_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`2"
 model/Conv1D_4/conv1d/ExpandDimsх
1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:`D*
dtype023
1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp
&model/Conv1D_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_4/conv1d/ExpandDims_1/dimѓ
"model/Conv1D_4/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:`D2$
"model/Conv1D_4/conv1d/ExpandDims_1ѓ
model/Conv1D_4/conv1dConv2D)model/Conv1D_4/conv1d/ExpandDims:output:0+model/Conv1D_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџD*
paddingVALID*
strides
2
model/Conv1D_4/conv1dП
model/Conv1D_4/conv1d/SqueezeSqueezemodel/Conv1D_4/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџD*
squeeze_dims

§џџџџџџџџ2
model/Conv1D_4/conv1d/SqueezeЙ
%model/Conv1D_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:D*
dtype02'
%model/Conv1D_4/BiasAdd/ReadVariableOpШ
model/Conv1D_4/BiasAddBiasAdd&model/Conv1D_4/conv1d/Squeeze:output:0-model/Conv1D_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџD2
model/Conv1D_4/BiasAdd
model/Conv1D_4/ReluRelumodel/Conv1D_4/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџD2
model/Conv1D_4/Relu{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
model/flatten/Const­
model/flatten/ReshapeReshape!model/Conv1D_4/Relu:activations:0model/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model/flatten/Reshape
model/dropout/IdentityIdentitymodel/flatten/Reshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model/dropout/IdentityИ
#model/Dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	S*
dtype02%
#model/Dense_1/MatMul/ReadVariableOpЖ
model/Dense_1/MatMulMatMulmodel/dropout/Identity:output:0+model/Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџS2
model/Dense_1/MatMulЖ
$model/Dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:S*
dtype02&
$model/Dense_1/BiasAdd/ReadVariableOpЙ
model/Dense_1/BiasAddBiasAddmodel/Dense_1/MatMul:product:0,model/Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџS2
model/Dense_1/BiasAdd
model/Dense_1/ReluRelumodel/Dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџS2
model/Dense_1/Relu
model/dropout_1/IdentityIdentity model/Dense_1/Relu:activations:0*
T0*'
_output_shapes
:џџџџџџџџџS2
model/dropout_1/IdentityЗ
#model/Dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:S!*
dtype02%
#model/Dense_2/MatMul/ReadVariableOpИ
model/Dense_2/MatMulMatMul!model/dropout_1/Identity:output:0+model/Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ!2
model/Dense_2/MatMulЖ
$model/Dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype02&
$model/Dense_2/BiasAdd/ReadVariableOpЙ
model/Dense_2/BiasAddBiasAddmodel/Dense_2/MatMul:product:0,model/Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ!2
model/Dense_2/BiasAdd
model/Dense_2/SoftmaxSoftmaxmodel/Dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ!2
model/Dense_2/Softmax§
IdentityIdentitymodel/Dense_2/Softmax:softmax:0&^model/Conv1D_1/BiasAdd/ReadVariableOp2^model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp&^model/Conv1D_2/BiasAdd/ReadVariableOp2^model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp&^model/Conv1D_3/BiasAdd/ReadVariableOp2^model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp&^model/Conv1D_4/BiasAdd/ReadVariableOp2^model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp%^model/Dense_1/BiasAdd/ReadVariableOp$^model/Dense_1/MatMul/ReadVariableOp%^model/Dense_2/BiasAdd/ReadVariableOp$^model/Dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџи: : : : : : : : : : : : 2N
%model/Conv1D_1/BiasAdd/ReadVariableOp%model/Conv1D_1/BiasAdd/ReadVariableOp2f
1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp2N
%model/Conv1D_2/BiasAdd/ReadVariableOp%model/Conv1D_2/BiasAdd/ReadVariableOp2f
1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp2N
%model/Conv1D_3/BiasAdd/ReadVariableOp%model/Conv1D_3/BiasAdd/ReadVariableOp2f
1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp2N
%model/Conv1D_4/BiasAdd/ReadVariableOp%model/Conv1D_4/BiasAdd/ReadVariableOp2f
1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp2L
$model/Dense_1/BiasAdd/ReadVariableOp$model/Dense_1/BiasAdd/ReadVariableOp2J
#model/Dense_1/MatMul/ReadVariableOp#model/Dense_1/MatMul/ReadVariableOp2L
$model/Dense_2/BiasAdd/ReadVariableOp$model/Dense_2/BiasAdd/ReadVariableOp2J
#model/Dense_2/MatMul/ReadVariableOp#model/Dense_2/MatMul/ReadVariableOp:U Q
,
_output_shapes
:џџџџџџџџџи
!
_user_specified_name	input_1
ѓ
`
B__inference_dropout_layer_call_and_return_conditional_losses_23877

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
м

C__inference_Conv1D_2_layer_call_and_return_conditional_losses_23802

inputsA
+conv1d_expanddims_1_readvariableop_resource:%0-
biasadd_readvariableop_resource:0
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ%2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:%0*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:%02
conv1d/ExpandDims_1И
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ0*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ0*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ02	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ02
ReluЉ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:џџџџџџџџџ02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ%: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџ%
 
_user_specified_nameinputs
А

(__inference_Conv1D_3_layer_call_fn_23836

inputs
unknown:0`
	unknown_0:`
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ!`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_230562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџ!`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ$0: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ$0
 
_user_specified_nameinputs
И

Б
#__inference_signature_wrapper_23525
input_1
unknown:%
	unknown_0:%
	unknown_1:%0
	unknown_2:0
	unknown_3:0`
	unknown_4:`
	unknown_5:`D
	unknown_6:D
	unknown_7:	S
	unknown_8:S
	unknown_9:S!

unknown_10:!
identityЂStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_229422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџи: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:џџџџџџџџџи
!
_user_specified_name	input_1
д

C__inference_Conv1D_4_layer_call_and_return_conditional_losses_23852

inputsA
+conv1d_expanddims_1_readvariableop_resource:`D-
biasadd_readvariableop_resource:D
identityЂBiasAdd/ReadVariableOpЂ"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2
conv1d/ExpandDims/dim
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:`D*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimЗ
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:`D2
conv1d/ExpandDims_1З
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџD*
paddingVALID*
strides
2
conv1d
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџD*
squeeze_dims

§џџџџџџџџ2
conv1d/Squeeze
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:D*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџD2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџD2
ReluЈ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:џџџџџџџџџD2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:џџџџџџџџџ`
 
_user_specified_nameinputs
к

Г
%__inference_model_layer_call_fn_23169
input_1
unknown:%
	unknown_0:%
	unknown_1:%0
	unknown_2:0
	unknown_3:0`
	unknown_4:`
	unknown_5:`D
	unknown_6:D
	unknown_7:	S
	unknown_8:S
	unknown_9:S!

unknown_10:!
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_231422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџи: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:џџџџџџџџџи
!
_user_specified_name	input_1
В

ѓ
B__inference_Dense_2_layer_call_and_return_conditional_losses_23957

inputs0
matmul_readvariableop_resource:S!-
biasadd_readvariableop_resource:!
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:S!*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ!2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:!*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ!2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ!2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџS: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџS
 
_user_specified_nameinputs
з

В
%__inference_model_layer_call_fn_23761

inputs
unknown:%
	unknown_0:%
	unknown_1:%0
	unknown_2:0
	unknown_3:0`
	unknown_4:`
	unknown_5:`D
	unknown_6:D
	unknown_7:	S
	unknown_8:S
	unknown_9:S!

unknown_10:!
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_233522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџи: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџи
 
_user_specified_nameinputs
а


@__inference_model_layer_call_and_return_conditional_losses_23703

inputsJ
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:%6
(conv1d_1_biasadd_readvariableop_resource:%J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:%06
(conv1d_2_biasadd_readvariableop_resource:0J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:0`6
(conv1d_3_biasadd_readvariableop_resource:`J
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:`D6
(conv1d_4_biasadd_readvariableop_resource:D9
&dense_1_matmul_readvariableop_resource:	S5
'dense_1_biasadd_readvariableop_resource:S8
&dense_2_matmul_readvariableop_resource:S!5
'dense_2_biasadd_readvariableop_resource:!
identityЂConv1D_1/BiasAdd/ReadVariableOpЂ+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpЂConv1D_2/BiasAdd/ReadVariableOpЂ+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpЂConv1D_3/BiasAdd/ReadVariableOpЂ+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOpЂConv1D_4/BiasAdd/ReadVariableOpЂ+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOpЂDense_1/BiasAdd/ReadVariableOpЂDense_1/MatMul/ReadVariableOpЂDense_2/BiasAdd/ReadVariableOpЂDense_2/MatMul/ReadVariableOp
Conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
Conv1D_1/conv1d/ExpandDims/dimВ
Conv1D_1/conv1d/ExpandDims
ExpandDimsinputs'Conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџи2
Conv1D_1/conv1d/ExpandDimsг
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:%*
dtype02-
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp
 Conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_1/conv1d/ExpandDims_1/dimл
Conv1D_1/conv1d/ExpandDims_1
ExpandDims3Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:%2
Conv1D_1/conv1d/ExpandDims_1м
Conv1D_1/conv1dConv2D#Conv1D_1/conv1d/ExpandDims:output:0%Conv1D_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџг%*
paddingVALID*
strides
2
Conv1D_1/conv1dЎ
Conv1D_1/conv1d/SqueezeSqueezeConv1D_1/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџг%*
squeeze_dims

§џџџџџџџџ2
Conv1D_1/conv1d/SqueezeЇ
Conv1D_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:%*
dtype02!
Conv1D_1/BiasAdd/ReadVariableOpБ
Conv1D_1/BiasAddBiasAdd Conv1D_1/conv1d/Squeeze:output:0'Conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџг%2
Conv1D_1/BiasAddx
Conv1D_1/ReluReluConv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџг%2
Conv1D_1/Relu
MaxPooling1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_1/ExpandDims/dimФ
MaxPooling1D_1/ExpandDims
ExpandDimsConv1D_1/Relu:activations:0&MaxPooling1D_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџг%2
MaxPooling1D_1/ExpandDimsЭ
MaxPooling1D_1/MaxPoolMaxPool"MaxPooling1D_1/ExpandDims:output:0*0
_output_shapes
:џџџџџџџџџ%*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_1/MaxPoolЊ
MaxPooling1D_1/SqueezeSqueezeMaxPooling1D_1/MaxPool:output:0*
T0*,
_output_shapes
:џџџџџџџџџ%*
squeeze_dims
2
MaxPooling1D_1/Squeeze
Conv1D_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
Conv1D_2/conv1d/ExpandDims/dimЫ
Conv1D_2/conv1d/ExpandDims
ExpandDimsMaxPooling1D_1/Squeeze:output:0'Conv1D_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ%2
Conv1D_2/conv1d/ExpandDimsг
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:%0*
dtype02-
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp
 Conv1D_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_2/conv1d/ExpandDims_1/dimл
Conv1D_2/conv1d/ExpandDims_1
ExpandDims3Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:%02
Conv1D_2/conv1d/ExpandDims_1м
Conv1D_2/conv1dConv2D#Conv1D_2/conv1d/ExpandDims:output:0%Conv1D_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:џџџџџџџџџ0*
paddingVALID*
strides
2
Conv1D_2/conv1dЎ
Conv1D_2/conv1d/SqueezeSqueezeConv1D_2/conv1d:output:0*
T0*,
_output_shapes
:џџџџџџџџџ0*
squeeze_dims

§џџџџџџџџ2
Conv1D_2/conv1d/SqueezeЇ
Conv1D_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02!
Conv1D_2/BiasAdd/ReadVariableOpБ
Conv1D_2/BiasAddBiasAdd Conv1D_2/conv1d/Squeeze:output:0'Conv1D_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ02
Conv1D_2/BiasAddx
Conv1D_2/ReluReluConv1D_2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ02
Conv1D_2/Relu
MaxPooling1D_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_2/ExpandDims/dimФ
MaxPooling1D_2/ExpandDims
ExpandDimsConv1D_2/Relu:activations:0&MaxPooling1D_2/ExpandDims/dim:output:0*
T0*0
_output_shapes
:џџџџџџџџџ02
MaxPooling1D_2/ExpandDimsЬ
MaxPooling1D_2/MaxPoolMaxPool"MaxPooling1D_2/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ$0*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_2/MaxPoolЉ
MaxPooling1D_2/SqueezeSqueezeMaxPooling1D_2/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ$0*
squeeze_dims
2
MaxPooling1D_2/Squeeze
Conv1D_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
Conv1D_3/conv1d/ExpandDims/dimЪ
Conv1D_3/conv1d/ExpandDims
ExpandDimsMaxPooling1D_2/Squeeze:output:0'Conv1D_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$02
Conv1D_3/conv1d/ExpandDimsг
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:0`*
dtype02-
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp
 Conv1D_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_3/conv1d/ExpandDims_1/dimл
Conv1D_3/conv1d/ExpandDims_1
ExpandDims3Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:0`2
Conv1D_3/conv1d/ExpandDims_1л
Conv1D_3/conv1dConv2D#Conv1D_3/conv1d/ExpandDims:output:0%Conv1D_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџ!`*
paddingVALID*
strides
2
Conv1D_3/conv1d­
Conv1D_3/conv1d/SqueezeSqueezeConv1D_3/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџ!`*
squeeze_dims

§џџџџџџџџ2
Conv1D_3/conv1d/SqueezeЇ
Conv1D_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype02!
Conv1D_3/BiasAdd/ReadVariableOpА
Conv1D_3/BiasAddBiasAdd Conv1D_3/conv1d/Squeeze:output:0'Conv1D_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ!`2
Conv1D_3/BiasAddw
Conv1D_3/ReluReluConv1D_3/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ!`2
Conv1D_3/Relu
MaxPooling1D_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_3/ExpandDims/dimУ
MaxPooling1D_3/ExpandDims
ExpandDimsConv1D_3/Relu:activations:0&MaxPooling1D_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ!`2
MaxPooling1D_3/ExpandDimsЬ
MaxPooling1D_3/MaxPoolMaxPool"MaxPooling1D_3/ExpandDims:output:0*/
_output_shapes
:џџџџџџџџџ`*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_3/MaxPoolЉ
MaxPooling1D_3/SqueezeSqueezeMaxPooling1D_3/MaxPool:output:0*
T0*+
_output_shapes
:џџџџџџџџџ`*
squeeze_dims
2
MaxPooling1D_3/Squeeze
Conv1D_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§џџџџџџџџ2 
Conv1D_4/conv1d/ExpandDims/dimЪ
Conv1D_4/conv1d/ExpandDims
ExpandDimsMaxPooling1D_3/Squeeze:output:0'Conv1D_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ`2
Conv1D_4/conv1d/ExpandDimsг
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:`D*
dtype02-
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp
 Conv1D_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_4/conv1d/ExpandDims_1/dimл
Conv1D_4/conv1d/ExpandDims_1
ExpandDims3Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:`D2
Conv1D_4/conv1d/ExpandDims_1л
Conv1D_4/conv1dConv2D#Conv1D_4/conv1d/ExpandDims:output:0%Conv1D_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:џџџџџџџџџD*
paddingVALID*
strides
2
Conv1D_4/conv1d­
Conv1D_4/conv1d/SqueezeSqueezeConv1D_4/conv1d:output:0*
T0*+
_output_shapes
:џџџџџџџџџD*
squeeze_dims

§џџџџџџџџ2
Conv1D_4/conv1d/SqueezeЇ
Conv1D_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:D*
dtype02!
Conv1D_4/BiasAdd/ReadVariableOpА
Conv1D_4/BiasAddBiasAdd Conv1D_4/conv1d/Squeeze:output:0'Conv1D_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџD2
Conv1D_4/BiasAddw
Conv1D_4/ReluReluConv1D_4/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџD2
Conv1D_4/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
flatten/Const
flatten/ReshapeReshapeConv1D_4/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ЁWЉ?2
dropout/dropout/Const
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/dropout/Mulv
dropout/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shapeо
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype0*
seed2џџџџ2.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fўy>2 
dropout/dropout/GreaterEqual/yп
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/dropout/Mul_1І
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	S*
dtype02
Dense_1/MatMul/ReadVariableOp
Dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџS2
Dense_1/MatMulЄ
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:S*
dtype02 
Dense_1/BiasAdd/ReadVariableOpЁ
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџS2
Dense_1/BiasAddp
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџS2
Dense_1/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *&дс?2
dropout_1/dropout/ConstЅ
dropout_1/dropout/MulMulDense_1/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџS2
dropout_1/dropout/Mul|
dropout_1/dropout/ShapeShapeDense_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shapeп
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџS*
dtype0*
seed220
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *8Ьн>2"
 dropout_1/dropout/GreaterEqual/yц
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџS2 
dropout_1/dropout/GreaterEqual
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:џџџџџџџџџS2
dropout_1/dropout/CastЂ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџS2
dropout_1/dropout/Mul_1Ѕ
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:S!*
dtype02
Dense_2/MatMul/ReadVariableOp 
Dense_2/MatMulMatMuldropout_1/dropout/Mul_1:z:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ!2
Dense_2/MatMulЄ
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype02 
Dense_2/BiasAdd/ReadVariableOpЁ
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ!2
Dense_2/BiasAddy
Dense_2/SoftmaxSoftmaxDense_2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ!2
Dense_2/SoftmaxЏ
IdentityIdentityDense_2/Softmax:softmax:0 ^Conv1D_1/BiasAdd/ReadVariableOp,^Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_2/BiasAdd/ReadVariableOp,^Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_3/BiasAdd/ReadVariableOp,^Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_4/BiasAdd/ReadVariableOp,^Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:џџџџџџџџџи: : : : : : : : : : : : 2B
Conv1D_1/BiasAdd/ReadVariableOpConv1D_1/BiasAdd/ReadVariableOp2Z
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_2/BiasAdd/ReadVariableOpConv1D_2/BiasAdd/ReadVariableOp2Z
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_3/BiasAdd/ReadVariableOpConv1D_3/BiasAdd/ReadVariableOp2Z
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp2B
Conv1D_4/BiasAdd/ReadVariableOpConv1D_4/BiasAdd/ReadVariableOp2Z
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp2@
Dense_1/BiasAdd/ReadVariableOpDense_1/BiasAdd/ReadVariableOp2>
Dense_1/MatMul/ReadVariableOpDense_1/MatMul/ReadVariableOp2@
Dense_2/BiasAdd/ReadVariableOpDense_2/BiasAdd/ReadVariableOp2>
Dense_2/MatMul/ReadVariableOpDense_2/MatMul/ReadVariableOp:T P
,
_output_shapes
:џџџџџџџџџи
 
_user_specified_nameinputs
А

(__inference_Conv1D_4_layer_call_fn_23861

inputs
unknown:`D
	unknown_0:D
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџD*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_230792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:џџџџџџџџџD2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ`: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:џџџџџџџџџ`
 
_user_specified_nameinputs"ЬL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Џ
serving_default
@
input_15
serving_default_input_1:0џџџџџџџџџи;
Dense_20
StatefulPartitionedCall:0џџџџџџџџџ!tensorflow/serving/predict:С
Ќp
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+Й&call_and_return_all_conditional_losses
К__call__
Л_default_save_signature"l
_tf_keras_networkњk{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 600, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "Conv1D_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 600, 1]}, "dtype": "float32", "filters": 37, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_1", "inbound_nodes": [[["Conv1D_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_2", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_2", "inbound_nodes": [[["MaxPooling1D_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_2", "inbound_nodes": [[["Conv1D_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3", "inbound_nodes": [[["MaxPooling1D_2", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_3", "inbound_nodes": [[["Conv1D_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_4", "trainable": true, "dtype": "float32", "filters": 68, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_4", "inbound_nodes": [[["MaxPooling1D_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["Conv1D_4", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.24413452260286977, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 83, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.43319870005964733, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["Dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 33, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Dense_2", 0, 0]]}, "shared_object_id": 25, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 600, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 600, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 600, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 600, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "Conv1D_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 600, 1]}, "dtype": "float32", "filters": 37, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_1", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_1", "inbound_nodes": [[["Conv1D_1", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Conv1D", "config": {"name": "Conv1D_2", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_2", "inbound_nodes": [[["MaxPooling1D_1", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_2", "inbound_nodes": [[["Conv1D_2", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3", "inbound_nodes": [[["MaxPooling1D_2", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_3", "inbound_nodes": [[["Conv1D_3", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Conv1D", "config": {"name": "Conv1D_4", "trainable": true, "dtype": "float32", "filters": 68, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_4", "inbound_nodes": [[["MaxPooling1D_3", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["Conv1D_4", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.24413452260286977, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 83, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_1", "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 20}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.43319870005964733, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["Dense_1", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 33, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_2", "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 24}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Dense_2", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 27}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0015300000086426735, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ѓ"№
_tf_keras_input_layerа{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 600, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 600, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ч

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+М&call_and_return_all_conditional_losses
Н__call__"Р

_tf_keras_layerІ
{"name": "Conv1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 600, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 600, 1]}, "dtype": "float32", "filters": 37, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 600, 1]}}
е
	variables
regularization_losses
trainable_variables
	keras_api
+О&call_and_return_all_conditional_losses
П__call__"Ф
_tf_keras_layerЊ{"name": "MaxPooling1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_1", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 29}}
ѕ


kernel
bias
 	variables
!regularization_losses
"trainable_variables
#	keras_api
+Р&call_and_return_all_conditional_losses
С__call__"Ю	
_tf_keras_layerД	{"name": "Conv1D_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_2", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["MaxPooling1D_1", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 37}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 148, 37]}}
е
$	variables
%regularization_losses
&trainable_variables
'	keras_api
+Т&call_and_return_all_conditional_losses
У__call__"Ф
_tf_keras_layerЊ{"name": "MaxPooling1D_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_2", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 31}}
і


(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
+Ф&call_and_return_all_conditional_losses
Х__call__"Я	
_tf_keras_layerЕ	{"name": "Conv1D_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_3", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["MaxPooling1D_2", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 48}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36, 48]}}
ж
.	variables
/regularization_losses
0trainable_variables
1	keras_api
+Ц&call_and_return_all_conditional_losses
Ч__call__"Х
_tf_keras_layerЋ{"name": "MaxPooling1D_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_3", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 33}}
і


2kernel
3bias
4	variables
5regularization_losses
6trainable_variables
7	keras_api
+Ш&call_and_return_all_conditional_losses
Щ__call__"Я	
_tf_keras_layerЕ	{"name": "Conv1D_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_4", "trainable": true, "dtype": "float32", "filters": 68, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["MaxPooling1D_3", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 96}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 96]}}
С
8	variables
9regularization_losses
:trainable_variables
;	keras_api
+Ъ&call_and_return_all_conditional_losses
Ы__call__"А
_tf_keras_layer{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_4", 0, 0, {}]]], "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 35}}
З
<	variables
=regularization_losses
>trainable_variables
?	keras_api
+Ь&call_and_return_all_conditional_losses
Э__call__"І
_tf_keras_layer{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.24413452260286977, "noise_shape": null, "seed": null}, "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 17}
	

@kernel
Abias
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
+Ю&call_and_return_all_conditional_losses
Я__call__"й
_tf_keras_layerП{"name": "Dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 83, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 408}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 408]}}
Л
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
+а&call_and_return_all_conditional_losses
б__call__"Њ
_tf_keras_layer{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.43319870005964733, "noise_shape": null, "seed": null}, "inbound_nodes": [[["Dense_1", 0, 0, {}]]], "shared_object_id": 21}
	

Jkernel
Kbias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
+в&call_and_return_all_conditional_losses
г__call__"м
_tf_keras_layerТ{"name": "Dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 33, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 83}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 83]}}
У
Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_ratemЁmЂmЃmЄ(mЅ)mІ2mЇ3mЈ@mЉAmЊJmЋKmЌv­vЎvЏvА(vБ)vВ2vГ3vД@vЕAvЖJvЗKvИ"
	optimizer
v
0
1
2
3
(4
)5
26
37
@8
A9
J10
K11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
(4
)5
26
37
@8
A9
J10
K11"
trackable_list_wrapper
Ю

Ulayers
Vlayer_regularization_losses
	variables
regularization_losses
Wnon_trainable_variables
trainable_variables
Xmetrics
Ylayer_metrics
К__call__
Л_default_save_signature
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
-
дserving_default"
signature_map
%:#%2Conv1D_1/kernel
:%2Conv1D_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А

Zlayers
[layer_regularization_losses
	variables
regularization_losses
\non_trainable_variables
trainable_variables
]metrics
^layer_metrics
Н__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А

_layers
`layer_regularization_losses
	variables
regularization_losses
anon_trainable_variables
trainable_variables
bmetrics
clayer_metrics
П__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
%:#%02Conv1D_2/kernel
:02Conv1D_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А

dlayers
elayer_regularization_losses
 	variables
!regularization_losses
fnon_trainable_variables
"trainable_variables
gmetrics
hlayer_metrics
С__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А

ilayers
jlayer_regularization_losses
$	variables
%regularization_losses
knon_trainable_variables
&trainable_variables
lmetrics
mlayer_metrics
У__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
%:#0`2Conv1D_3/kernel
:`2Conv1D_3/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
А

nlayers
olayer_regularization_losses
*	variables
+regularization_losses
pnon_trainable_variables
,trainable_variables
qmetrics
rlayer_metrics
Х__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А

slayers
tlayer_regularization_losses
.	variables
/regularization_losses
unon_trainable_variables
0trainable_variables
vmetrics
wlayer_metrics
Ч__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
%:#`D2Conv1D_4/kernel
:D2Conv1D_4/bias
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
А

xlayers
ylayer_regularization_losses
4	variables
5regularization_losses
znon_trainable_variables
6trainable_variables
{metrics
|layer_metrics
Щ__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В

}layers
~layer_regularization_losses
8	variables
9regularization_losses
non_trainable_variables
:trainable_variables
metrics
layer_metrics
Ы__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layers
 layer_regularization_losses
<	variables
=regularization_losses
non_trainable_variables
>trainable_variables
metrics
layer_metrics
Э__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
!:	S2Dense_1/kernel
:S2Dense_1/bias
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
Е
layers
 layer_regularization_losses
B	variables
Cregularization_losses
non_trainable_variables
Dtrainable_variables
metrics
layer_metrics
Я__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
layers
 layer_regularization_losses
F	variables
Gregularization_losses
non_trainable_variables
Htrainable_variables
metrics
layer_metrics
б__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 :S!2Dense_2/kernel
:!2Dense_2/bias
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
Е
layers
 layer_regularization_losses
L	variables
Mregularization_losses
non_trainable_variables
Ntrainable_variables
metrics
layer_metrics
г__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
и

total

count
	variables
	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 38}


total

count

_fn_kwargs
	variables
 	keras_api"а
_tf_keras_metricЕ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 27}
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
*:(%2Adam/Conv1D_1/kernel/m
 :%2Adam/Conv1D_1/bias/m
*:(%02Adam/Conv1D_2/kernel/m
 :02Adam/Conv1D_2/bias/m
*:(0`2Adam/Conv1D_3/kernel/m
 :`2Adam/Conv1D_3/bias/m
*:(`D2Adam/Conv1D_4/kernel/m
 :D2Adam/Conv1D_4/bias/m
&:$	S2Adam/Dense_1/kernel/m
:S2Adam/Dense_1/bias/m
%:#S!2Adam/Dense_2/kernel/m
:!2Adam/Dense_2/bias/m
*:(%2Adam/Conv1D_1/kernel/v
 :%2Adam/Conv1D_1/bias/v
*:(%02Adam/Conv1D_2/kernel/v
 :02Adam/Conv1D_2/bias/v
*:(0`2Adam/Conv1D_3/kernel/v
 :`2Adam/Conv1D_3/bias/v
*:(`D2Adam/Conv1D_4/kernel/v
 :D2Adam/Conv1D_4/bias/v
&:$	S2Adam/Dense_1/kernel/v
:S2Adam/Dense_1/bias/v
%:#S!2Adam/Dense_2/kernel/v
:!2Adam/Dense_2/bias/v
Ю2Ы
@__inference_model_layer_call_and_return_conditional_losses_23607
@__inference_model_layer_call_and_return_conditional_losses_23703
@__inference_model_layer_call_and_return_conditional_losses_23448
@__inference_model_layer_call_and_return_conditional_losses_23488Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
т2п
%__inference_model_layer_call_fn_23169
%__inference_model_layer_call_fn_23732
%__inference_model_layer_call_fn_23761
%__inference_model_layer_call_fn_23408Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
у2р
 __inference__wrapped_model_22942Л
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *+Ђ(
&#
input_1џџџџџџџџџи
э2ъ
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_23777Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_Conv1D_1_layer_call_fn_23786Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Є2Ё
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_22951г
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
.__inference_MaxPooling1D_1_layer_call_fn_22957г
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
э2ъ
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_23802Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_Conv1D_2_layer_call_fn_23811Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Є2Ё
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_22966г
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
.__inference_MaxPooling1D_2_layer_call_fn_22972г
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
э2ъ
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_23827Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_Conv1D_3_layer_call_fn_23836Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Є2Ё
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_22981г
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
.__inference_MaxPooling1D_3_layer_call_fn_22987г
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
э2ъ
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_23852Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_Conv1D_4_layer_call_fn_23861Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_flatten_layer_call_and_return_conditional_losses_23867Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_flatten_layer_call_fn_23872Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Т2П
B__inference_dropout_layer_call_and_return_conditional_losses_23877
B__inference_dropout_layer_call_and_return_conditional_losses_23889Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
'__inference_dropout_layer_call_fn_23894
'__inference_dropout_layer_call_fn_23899Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ь2щ
B__inference_Dense_1_layer_call_and_return_conditional_losses_23910Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_Dense_1_layer_call_fn_23919Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ц2У
D__inference_dropout_1_layer_call_and_return_conditional_losses_23924
D__inference_dropout_1_layer_call_and_return_conditional_losses_23936Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
)__inference_dropout_1_layer_call_fn_23941
)__inference_dropout_1_layer_call_fn_23946Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ь2щ
B__inference_Dense_2_layer_call_and_return_conditional_losses_23957Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_Dense_2_layer_call_fn_23966Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЪBЧ
#__inference_signature_wrapper_23525input_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ­
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_23777f4Ђ1
*Ђ'
%"
inputsџџџџџџџџџи
Њ "*Ђ'
 
0џџџџџџџџџг%
 
(__inference_Conv1D_1_layer_call_fn_23786Y4Ђ1
*Ђ'
%"
inputsџџџџџџџџџи
Њ "џџџџџџџџџг%­
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_23802f4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ%
Њ "*Ђ'
 
0џџџџџџџџџ0
 
(__inference_Conv1D_2_layer_call_fn_23811Y4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ%
Њ "џџџџџџџџџ0Ћ
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_23827d()3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ$0
Њ ")Ђ&

0џџџџџџџџџ!`
 
(__inference_Conv1D_3_layer_call_fn_23836W()3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ$0
Њ "џџџџџџџџџ!`Ћ
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_23852d233Ђ0
)Ђ&
$!
inputsџџџџџџџџџ`
Њ ")Ђ&

0џџџџџџџџџD
 
(__inference_Conv1D_4_layer_call_fn_23861W233Ђ0
)Ђ&
$!
inputsџџџџџџџџџ`
Њ "џџџџџџџџџDЃ
B__inference_Dense_1_layer_call_and_return_conditional_losses_23910]@A0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџS
 {
'__inference_Dense_1_layer_call_fn_23919P@A0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџSЂ
B__inference_Dense_2_layer_call_and_return_conditional_losses_23957\JK/Ђ,
%Ђ"
 
inputsџџџџџџџџџS
Њ "%Ђ"

0џџџџџџџџџ!
 z
'__inference_Dense_2_layer_call_fn_23966OJK/Ђ,
%Ђ"
 
inputsџџџџџџџџџS
Њ "џџџџџџџџџ!в
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_22951EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";Ђ8
1.
0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Љ
.__inference_MaxPooling1D_1_layer_call_fn_22957wEЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".+'џџџџџџџџџџџџџџџџџџџџџџџџџџџв
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_22966EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";Ђ8
1.
0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Љ
.__inference_MaxPooling1D_2_layer_call_fn_22972wEЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".+'џџџџџџџџџџџџџџџџџџџџџџџџџџџв
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_22981EЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";Ђ8
1.
0'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Љ
.__inference_MaxPooling1D_3_layer_call_fn_22987wEЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 __inference__wrapped_model_22942x()23@AJK5Ђ2
+Ђ(
&#
input_1џџџџџџџџџи
Њ "1Њ.
,
Dense_2!
Dense_2џџџџџџџџџ!Є
D__inference_dropout_1_layer_call_and_return_conditional_losses_23924\3Ђ0
)Ђ&
 
inputsџџџџџџџџџS
p 
Њ "%Ђ"

0џџџџџџџџџS
 Є
D__inference_dropout_1_layer_call_and_return_conditional_losses_23936\3Ђ0
)Ђ&
 
inputsџџџџџџџџџS
p
Њ "%Ђ"

0џџџџџџџџџS
 |
)__inference_dropout_1_layer_call_fn_23941O3Ђ0
)Ђ&
 
inputsџџџџџџџџџS
p 
Њ "џџџџџџџџџS|
)__inference_dropout_1_layer_call_fn_23946O3Ђ0
)Ђ&
 
inputsџџџџџџџџџS
p
Њ "џџџџџџџџџSЄ
B__inference_dropout_layer_call_and_return_conditional_losses_23877^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "&Ђ#

0џџџџџџџџџ
 Є
B__inference_dropout_layer_call_and_return_conditional_losses_23889^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "&Ђ#

0џџџџџџџџџ
 |
'__inference_dropout_layer_call_fn_23894Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ|
'__inference_dropout_layer_call_fn_23899Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџЃ
B__inference_flatten_layer_call_and_return_conditional_losses_23867]3Ђ0
)Ђ&
$!
inputsџџџџџџџџџD
Њ "&Ђ#

0џџџџџџџџџ
 {
'__inference_flatten_layer_call_fn_23872P3Ђ0
)Ђ&
$!
inputsџџџџџџџџџD
Њ "џџџџџџџџџИ
@__inference_model_layer_call_and_return_conditional_losses_23448t()23@AJK=Ђ:
3Ђ0
&#
input_1џџџџџџџџџи
p 

 
Њ "%Ђ"

0џџџџџџџџџ!
 И
@__inference_model_layer_call_and_return_conditional_losses_23488t()23@AJK=Ђ:
3Ђ0
&#
input_1џџџџџџџџџи
p

 
Њ "%Ђ"

0џџџџџџџџџ!
 З
@__inference_model_layer_call_and_return_conditional_losses_23607s()23@AJK<Ђ9
2Ђ/
%"
inputsџџџџџџџџџи
p 

 
Њ "%Ђ"

0џџџџџџџџџ!
 З
@__inference_model_layer_call_and_return_conditional_losses_23703s()23@AJK<Ђ9
2Ђ/
%"
inputsџџџџџџџџџи
p

 
Њ "%Ђ"

0џџџџџџџџџ!
 
%__inference_model_layer_call_fn_23169g()23@AJK=Ђ:
3Ђ0
&#
input_1џџџџџџџџџи
p 

 
Њ "џџџџџџџџџ!
%__inference_model_layer_call_fn_23408g()23@AJK=Ђ:
3Ђ0
&#
input_1џџџџџџџџџи
p

 
Њ "џџџџџџџџџ!
%__inference_model_layer_call_fn_23732f()23@AJK<Ђ9
2Ђ/
%"
inputsџџџџџџџџџи
p 

 
Њ "џџџџџџџџџ!
%__inference_model_layer_call_fn_23761f()23@AJK<Ђ9
2Ђ/
%"
inputsџџџџџџџџџи
p

 
Њ "џџџџџџџџџ!Ћ
#__inference_signature_wrapper_23525()23@AJK@Ђ=
Ђ 
6Њ3
1
input_1&#
input_1џџџџџџџџџи"1Њ.
,
Dense_2!
Dense_2џџџџџџџџџ!