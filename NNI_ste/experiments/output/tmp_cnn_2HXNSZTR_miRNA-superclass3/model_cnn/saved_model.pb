??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
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
?
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
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718ޙ
~
Conv1D_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameConv1D_1/kernel
w
#Conv1D_1/kernel/Read/ReadVariableOpReadVariableOpConv1D_1/kernel*"
_output_shapes
:	*
dtype0
r
Conv1D_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameConv1D_1/bias
k
!Conv1D_1/bias/Read/ReadVariableOpReadVariableOpConv1D_1/bias*
_output_shapes
:	*
dtype0
~
Conv1D_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameConv1D_2/kernel
w
#Conv1D_2/kernel/Read/ReadVariableOpReadVariableOpConv1D_2/kernel*"
_output_shapes
:	*
dtype0
r
Conv1D_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv1D_2/bias
k
!Conv1D_2/bias/Read/ReadVariableOpReadVariableOpConv1D_2/bias*
_output_shapes
:*
dtype0
~
Conv1D_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:.* 
shared_nameConv1D_3/kernel
w
#Conv1D_3/kernel/Read/ReadVariableOpReadVariableOpConv1D_3/kernel*"
_output_shapes
:.*
dtype0
r
Conv1D_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*
shared_nameConv1D_3/bias
k
!Conv1D_3/bias/Read/ReadVariableOpReadVariableOpConv1D_3/bias*
_output_shapes
:.*
dtype0
~
Conv1D_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:.* 
shared_nameConv1D_4/kernel
w
#Conv1D_4/kernel/Read/ReadVariableOpReadVariableOpConv1D_4/kernel*"
_output_shapes
:.*
dtype0
r
Conv1D_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameConv1D_4/bias
k
!Conv1D_4/bias/Read/ReadVariableOpReadVariableOpConv1D_4/bias*
_output_shapes
:*
dtype0
x
Dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:~t*
shared_nameDense_1/kernel
q
"Dense_1/kernel/Read/ReadVariableOpReadVariableOpDense_1/kernel*
_output_shapes

:~t*
dtype0
p
Dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:t*
shared_nameDense_1/bias
i
 Dense_1/bias/Read/ReadVariableOpReadVariableOpDense_1/bias*
_output_shapes
:t*
dtype0
x
Dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:t!*
shared_nameDense_2/kernel
q
"Dense_2/kernel/Read/ReadVariableOpReadVariableOpDense_2/kernel*
_output_shapes

:t!*
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
?
Adam/Conv1D_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/Conv1D_1/kernel/m
?
*Adam/Conv1D_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/kernel/m*"
_output_shapes
:	*
dtype0
?
Adam/Conv1D_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/Conv1D_1/bias/m
y
(Adam/Conv1D_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/bias/m*
_output_shapes
:	*
dtype0
?
Adam/Conv1D_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/Conv1D_2/kernel/m
?
*Adam/Conv1D_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/kernel/m*"
_output_shapes
:	*
dtype0
?
Adam/Conv1D_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv1D_2/bias/m
y
(Adam/Conv1D_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/bias/m*
_output_shapes
:*
dtype0
?
Adam/Conv1D_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*'
shared_nameAdam/Conv1D_3/kernel/m
?
*Adam/Conv1D_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/kernel/m*"
_output_shapes
:.*
dtype0
?
Adam/Conv1D_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*%
shared_nameAdam/Conv1D_3/bias/m
y
(Adam/Conv1D_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/bias/m*
_output_shapes
:.*
dtype0
?
Adam/Conv1D_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*'
shared_nameAdam/Conv1D_4/kernel/m
?
*Adam/Conv1D_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/kernel/m*"
_output_shapes
:.*
dtype0
?
Adam/Conv1D_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv1D_4/bias/m
y
(Adam/Conv1D_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/bias/m*
_output_shapes
:*
dtype0
?
Adam/Dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:~t*&
shared_nameAdam/Dense_1/kernel/m

)Adam/Dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_1/kernel/m*
_output_shapes

:~t*
dtype0
~
Adam/Dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:t*$
shared_nameAdam/Dense_1/bias/m
w
'Adam/Dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/Dense_1/bias/m*
_output_shapes
:t*
dtype0
?
Adam/Dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:t!*&
shared_nameAdam/Dense_2/kernel/m

)Adam/Dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Dense_2/kernel/m*
_output_shapes

:t!*
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
?
Adam/Conv1D_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/Conv1D_1/kernel/v
?
*Adam/Conv1D_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/kernel/v*"
_output_shapes
:	*
dtype0
?
Adam/Conv1D_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/Conv1D_1/bias/v
y
(Adam/Conv1D_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_1/bias/v*
_output_shapes
:	*
dtype0
?
Adam/Conv1D_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/Conv1D_2/kernel/v
?
*Adam/Conv1D_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/kernel/v*"
_output_shapes
:	*
dtype0
?
Adam/Conv1D_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv1D_2/bias/v
y
(Adam/Conv1D_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_2/bias/v*
_output_shapes
:*
dtype0
?
Adam/Conv1D_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*'
shared_nameAdam/Conv1D_3/kernel/v
?
*Adam/Conv1D_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/kernel/v*"
_output_shapes
:.*
dtype0
?
Adam/Conv1D_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*%
shared_nameAdam/Conv1D_3/bias/v
y
(Adam/Conv1D_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_3/bias/v*
_output_shapes
:.*
dtype0
?
Adam/Conv1D_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:.*'
shared_nameAdam/Conv1D_4/kernel/v
?
*Adam/Conv1D_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/kernel/v*"
_output_shapes
:.*
dtype0
?
Adam/Conv1D_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/Conv1D_4/bias/v
y
(Adam/Conv1D_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/Conv1D_4/bias/v*
_output_shapes
:*
dtype0
?
Adam/Dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:~t*&
shared_nameAdam/Dense_1/kernel/v

)Adam/Dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_1/kernel/v*
_output_shapes

:~t*
dtype0
~
Adam/Dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:t*$
shared_nameAdam/Dense_1/bias/v
w
'Adam/Dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/Dense_1/bias/v*
_output_shapes
:t*
dtype0
?
Adam/Dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:t!*&
shared_nameAdam/Dense_2/kernel/v

)Adam/Dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Dense_2/kernel/v*
_output_shapes

:t!*
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
?P
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?O
value?OB?O B?O
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
 regularization_losses
!	variables
"	keras_api
h

#kernel
$bias
%trainable_variables
&regularization_losses
'	variables
(	keras_api
R
)trainable_variables
*regularization_losses
+	variables
,	keras_api
R
-trainable_variables
.regularization_losses
/	variables
0	keras_api
h

1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
R
7trainable_variables
8regularization_losses
9	variables
:	keras_api
R
;trainable_variables
<regularization_losses
=	variables
>	keras_api
h

?kernel
@bias
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
R
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
h

Ikernel
Jbias
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
h

Okernel
Pbias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
?
Uiter

Vbeta_1

Wbeta_2
	Xdecay
Ylearning_ratem?m?#m?$m?1m?2m??m?@m?Im?Jm?Om?Pm?v?v?#v?$v?1v?2v??v?@v?Iv?Jv?Ov?Pv?
 
V
0
1
#2
$3
14
25
?6
@7
I8
J9
O10
P11
V
0
1
#2
$3
14
25
?6
@7
I8
J9
O10
P11
?
Zmetrics
regularization_losses

[layers
\non_trainable_variables
trainable_variables
]layer_metrics
	variables
^layer_regularization_losses
 
[Y
VARIABLE_VALUEConv1D_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
_metrics

`layers
anon_trainable_variables
trainable_variables
regularization_losses
blayer_metrics
	variables
clayer_regularization_losses
 
 
 
?
dmetrics

elayers
fnon_trainable_variables
trainable_variables
regularization_losses
glayer_metrics
	variables
hlayer_regularization_losses
 
 
 
?
imetrics

jlayers
knon_trainable_variables
trainable_variables
 regularization_losses
llayer_metrics
!	variables
mlayer_regularization_losses
[Y
VARIABLE_VALUEConv1D_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
?
nmetrics

olayers
pnon_trainable_variables
%trainable_variables
&regularization_losses
qlayer_metrics
'	variables
rlayer_regularization_losses
 
 
 
?
smetrics

tlayers
unon_trainable_variables
)trainable_variables
*regularization_losses
vlayer_metrics
+	variables
wlayer_regularization_losses
 
 
 
?
xmetrics

ylayers
znon_trainable_variables
-trainable_variables
.regularization_losses
{layer_metrics
/	variables
|layer_regularization_losses
[Y
VARIABLE_VALUEConv1D_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21
 

10
21
?
}metrics

~layers
non_trainable_variables
3trainable_variables
4regularization_losses
?layer_metrics
5	variables
 ?layer_regularization_losses
 
 
 
?
?metrics
?layers
?non_trainable_variables
7trainable_variables
8regularization_losses
?layer_metrics
9	variables
 ?layer_regularization_losses
 
 
 
?
?metrics
?layers
?non_trainable_variables
;trainable_variables
<regularization_losses
?layer_metrics
=	variables
 ?layer_regularization_losses
[Y
VARIABLE_VALUEConv1D_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEConv1D_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
@1
 

?0
@1
?
?metrics
?layers
?non_trainable_variables
Atrainable_variables
Bregularization_losses
?layer_metrics
C	variables
 ?layer_regularization_losses
 
 
 
?
?metrics
?layers
?non_trainable_variables
Etrainable_variables
Fregularization_losses
?layer_metrics
G	variables
 ?layer_regularization_losses
ZX
VARIABLE_VALUEDense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1
 

I0
J1
?
?metrics
?layers
?non_trainable_variables
Ktrainable_variables
Lregularization_losses
?layer_metrics
M	variables
 ?layer_regularization_losses
ZX
VARIABLE_VALUEDense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEDense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

O0
P1
 

O0
P1
?
?metrics
?layers
?non_trainable_variables
Qtrainable_variables
Rregularization_losses
?layer_metrics
S	variables
 ?layer_regularization_losses
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

?0
?1
f
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
13
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
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
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
?
serving_default_input_1Placeholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Conv1D_1/kernelConv1D_1/biasConv1D_2/kernelConv1D_2/biasConv1D_3/kernelConv1D_3/biasConv1D_4/kernelConv1D_4/biasDense_1/kernelDense_1/biasDense_2/kernelDense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_24623
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
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
GPU2*0J 8? *'
f"R 
__inference__traced_save_25258
?	
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
GPU2*0J 8? **
f%R#
!__inference__traced_restore_25403??

?;
?
@__inference_model_layer_call_and_return_conditional_losses_24586
input_1$
conv1d_1_24548:	
conv1d_1_24550:	$
conv1d_2_24555:	
conv1d_2_24557:$
conv1d_3_24562:.
conv1d_3_24564:.$
conv1d_4_24569:.
conv1d_4_24571:
dense_1_24575:~t
dense_1_24577:t
dense_2_24580:t!
dense_2_24582:!
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall? Conv1D_3/StatefulPartitionedCall? Conv1D_4/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1_24548conv1d_1_24550*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_240752"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????J	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_240162 
MaxPooling1D_1/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????J	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_243632!
dropout/StatefulPartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv1d_2_24555conv1d_2_24557*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????G*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_241052"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_240312 
MaxPooling1D_2/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_243302#
!dropout_1/StatefulPartitionedCall?
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv1d_3_24562conv1d_3_24564*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????.*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_241352"
 Conv1D_3/StatefulPartitionedCall?
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_240462 
MaxPooling1D_3/PartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_242972#
!dropout_2/StatefulPartitionedCall?
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv1d_4_24569conv1d_4_24571*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_241652"
 Conv1D_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????~* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_241772
flatten/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_24575dense_1_24577*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????t*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_241902!
Dense_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_24580dense_2_24582*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_242072!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?6
?
@__inference_model_layer_call_and_return_conditional_losses_24214

inputs$
conv1d_1_24076:	
conv1d_1_24078:	$
conv1d_2_24106:	
conv1d_2_24108:$
conv1d_3_24136:.
conv1d_3_24138:.$
conv1d_4_24166:.
conv1d_4_24168:
dense_1_24191:~t
dense_1_24193:t
dense_2_24208:t!
dense_2_24210:!
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall? Conv1D_3/StatefulPartitionedCall? Conv1D_4/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1_24076conv1d_1_24078*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_240752"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????J	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_240162 
MaxPooling1D_1/PartitionedCall?
dropout/PartitionedCallPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????J	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_240872
dropout/PartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv1d_2_24106conv1d_2_24108*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????G*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_241052"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_240312 
MaxPooling1D_2/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_241172
dropout_1/PartitionedCall?
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv1d_3_24136conv1d_3_24138*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????.*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_241352"
 Conv1D_3/StatefulPartitionedCall?
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_240462 
MaxPooling1D_3/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_241472
dropout_2/PartitionedCall?
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv1d_4_24166conv1d_4_24168*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_241652"
 Conv1D_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????~* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_241772
flatten/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_24191dense_1_24193*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????t*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_241902!
Dense_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_24208dense_2_24210*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_242072!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_Dense_1_layer_call_fn_25080

inputs
unknown:~t
	unknown_0:t
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????t*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_241902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????t2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????~: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????~
 
_user_specified_nameinputs
?
?
(__inference_Conv1D_1_layer_call_fn_24893

inputs
unknown:	
	unknown_0:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_240752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_MaxPooling1D_3_layer_call_fn_24052

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_240462
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
E
)__inference_dropout_1_layer_call_fn_24967

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_241172
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????#:S O
+
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_25055

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????~   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????~2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_25060

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????~* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_241772
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_Dense_2_layer_call_and_return_conditional_losses_24207

inputs0
matmul_readvariableop_resource:t!-
biasadd_readvariableop_resource:!
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:t!*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:!*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????!2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????t: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????t
 
_user_specified_nameinputs
?6
?
@__inference_model_layer_call_and_return_conditional_losses_24545
input_1$
conv1d_1_24507:	
conv1d_1_24509:	$
conv1d_2_24514:	
conv1d_2_24516:$
conv1d_3_24521:.
conv1d_3_24523:.$
conv1d_4_24528:.
conv1d_4_24530:
dense_1_24534:~t
dense_1_24536:t
dense_2_24539:t!
dense_2_24541:!
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall? Conv1D_3/StatefulPartitionedCall? Conv1D_4/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1_24507conv1d_1_24509*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_240752"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????J	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_240162 
MaxPooling1D_1/PartitionedCall?
dropout/PartitionedCallPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????J	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_240872
dropout/PartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0conv1d_2_24514conv1d_2_24516*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????G*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_241052"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_240312 
MaxPooling1D_2/PartitionedCall?
dropout_1/PartitionedCallPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_241172
dropout_1/PartitionedCall?
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv1d_3_24521conv1d_3_24523*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????.*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_241352"
 Conv1D_3/StatefulPartitionedCall?
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_240462 
MaxPooling1D_3/PartitionedCall?
dropout_2/PartitionedCallPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_241472
dropout_2/PartitionedCall?
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv1d_4_24528conv1d_4_24530*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_241652"
 Conv1D_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????~* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_241772
flatten/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_24534dense_1_24536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????t*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_241902!
Dense_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_24539dense_2_24541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_242072!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?;
?
@__inference_model_layer_call_and_return_conditional_losses_24448

inputs$
conv1d_1_24410:	
conv1d_1_24412:	$
conv1d_2_24417:	
conv1d_2_24419:$
conv1d_3_24424:.
conv1d_3_24426:.$
conv1d_4_24431:.
conv1d_4_24433:
dense_1_24437:~t
dense_1_24439:t
dense_2_24442:t!
dense_2_24444:!
identity?? Conv1D_1/StatefulPartitionedCall? Conv1D_2/StatefulPartitionedCall? Conv1D_3/StatefulPartitionedCall? Conv1D_4/StatefulPartitionedCall?Dense_1/StatefulPartitionedCall?Dense_2/StatefulPartitionedCall?dropout/StatefulPartitionedCall?!dropout_1/StatefulPartitionedCall?!dropout_2/StatefulPartitionedCall?
 Conv1D_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1_24410conv1d_1_24412*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_240752"
 Conv1D_1/StatefulPartitionedCall?
MaxPooling1D_1/PartitionedCallPartitionedCall)Conv1D_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????J	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_240162 
MaxPooling1D_1/PartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????J	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_243632!
dropout/StatefulPartitionedCall?
 Conv1D_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0conv1d_2_24417conv1d_2_24419*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????G*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_241052"
 Conv1D_2/StatefulPartitionedCall?
MaxPooling1D_2/PartitionedCallPartitionedCall)Conv1D_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_240312 
MaxPooling1D_2/PartitionedCall?
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_2/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_243302#
!dropout_1/StatefulPartitionedCall?
 Conv1D_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv1d_3_24424conv1d_3_24426*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????.*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_241352"
 Conv1D_3/StatefulPartitionedCall?
MaxPooling1D_3/PartitionedCallPartitionedCall)Conv1D_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_240462 
MaxPooling1D_3/PartitionedCall?
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall'MaxPooling1D_3/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_242972#
!dropout_2/StatefulPartitionedCall?
 Conv1D_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv1d_4_24431conv1d_4_24433*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_241652"
 Conv1D_4/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)Conv1D_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????~* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_241772
flatten/PartitionedCall?
Dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_24437dense_1_24439*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????t*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_1_layer_call_and_return_conditional_losses_241902!
Dense_1/StatefulPartitionedCall?
Dense_2/StatefulPartitionedCallStatefulPartitionedCall(Dense_1/StatefulPartitionedCall:output:0dense_2_24442dense_2_24444*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_242072!
Dense_2/StatefulPartitionedCall?
IdentityIdentity(Dense_2/StatefulPartitionedCall:output:0!^Conv1D_1/StatefulPartitionedCall!^Conv1D_2/StatefulPartitionedCall!^Conv1D_3/StatefulPartitionedCall!^Conv1D_4/StatefulPartitionedCall ^Dense_1/StatefulPartitionedCall ^Dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2D
 Conv1D_1/StatefulPartitionedCall Conv1D_1/StatefulPartitionedCall2D
 Conv1D_2/StatefulPartitionedCall Conv1D_2/StatefulPartitionedCall2D
 Conv1D_3/StatefulPartitionedCall Conv1D_3/StatefulPartitionedCall2D
 Conv1D_4/StatefulPartitionedCall Conv1D_4/StatefulPartitionedCall2B
Dense_1/StatefulPartitionedCallDense_1/StatefulPartitionedCall2B
Dense_2/StatefulPartitionedCallDense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_24241
input_1
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:.
	unknown_4:.
	unknown_5:.
	unknown_6:
	unknown_7:~t
	unknown_8:t
	unknown_9:t!

unknown_10:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_242142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?

?
%__inference_model_layer_call_fn_24839

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:.
	unknown_4:.
	unknown_5:.
	unknown_6:
	unknown_7:~t
	unknown_8:t
	unknown_9:t!

unknown_10:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_242142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?\
?
__inference__traced_save_25258
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

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop*savev2_conv1d_3_kernel_read_readvariableop(savev2_conv1d_3_bias_read_readvariableop*savev2_conv1d_4_kernel_read_readvariableop(savev2_conv1d_4_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop1savev2_adam_conv1d_2_kernel_m_read_readvariableop/savev2_adam_conv1d_2_bias_m_read_readvariableop1savev2_adam_conv1d_3_kernel_m_read_readvariableop/savev2_adam_conv1d_3_bias_m_read_readvariableop1savev2_adam_conv1d_4_kernel_m_read_readvariableop/savev2_adam_conv1d_4_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop1savev2_adam_conv1d_2_kernel_v_read_readvariableop/savev2_adam_conv1d_2_bias_v_read_readvariableop1savev2_adam_conv1d_3_kernel_v_read_readvariableop/savev2_adam_conv1d_3_bias_v_read_readvariableop1savev2_adam_conv1d_4_kernel_v_read_readvariableop/savev2_adam_conv1d_4_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *<
dtypes2
02.	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :	:	:	::.:.:.::~t:t:t!:!: : : : : : : : : :	:	:	::.:.:.::~t:t:t!:!:	:	:	::.:.:.::~t:t:t!:!: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:	: 

_output_shapes
:	:($
"
_output_shapes
:	: 

_output_shapes
::($
"
_output_shapes
:.: 

_output_shapes
:.:($
"
_output_shapes
:.: 

_output_shapes
::$	 

_output_shapes

:~t: 


_output_shapes
:t:$ 

_output_shapes

:t!: 
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
:	: 

_output_shapes
:	:($
"
_output_shapes
:	: 

_output_shapes
::($
"
_output_shapes
:.: 

_output_shapes
:.:($
"
_output_shapes
:.: 

_output_shapes
::$ 

_output_shapes

:~t: 

_output_shapes
:t:$  

_output_shapes

:t!: !

_output_shapes
:!:("$
"
_output_shapes
:	: #

_output_shapes
:	:($$
"
_output_shapes
:	: %

_output_shapes
::(&$
"
_output_shapes
:.: '

_output_shapes
:.:(($
"
_output_shapes
:.: )

_output_shapes
::$* 

_output_shapes

:~t: +

_output_shapes
:t:$, 

_output_shapes

:t!: -

_output_shapes
:!:.

_output_shapes
: 
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_24147

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????.2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????.2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????.:S O
+
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
b
)__inference_dropout_2_layer_call_fn_25024

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_242972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????.2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????.22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????.
 
_user_specified_nameinputs
ˍ
?

@__inference_model_layer_call_and_return_conditional_losses_24810

inputsJ
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:	6
(conv1d_1_biasadd_readvariableop_resource:	J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:	6
(conv1d_2_biasadd_readvariableop_resource:J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:.6
(conv1d_3_biasadd_readvariableop_resource:.J
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:.6
(conv1d_4_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:~t5
'dense_1_biasadd_readvariableop_resource:t8
&dense_2_matmul_readvariableop_resource:t!5
'dense_2_biasadd_readvariableop_resource:!
identity??Conv1D_1/BiasAdd/ReadVariableOp?+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_2/BiasAdd/ReadVariableOp?+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_3/BiasAdd/ReadVariableOp?+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_4/BiasAdd/ReadVariableOp?+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?Dense_1/BiasAdd/ReadVariableOp?Dense_1/MatMul/ReadVariableOp?Dense_2/BiasAdd/ReadVariableOp?Dense_2/MatMul/ReadVariableOp?
Conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_1/conv1d/ExpandDims/dim?
Conv1D_1/conv1d/ExpandDims
ExpandDimsinputs'Conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
Conv1D_1/conv1d/ExpandDims?
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	*
dtype02-
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_1/conv1d/ExpandDims_1/dim?
Conv1D_1/conv1d/ExpandDims_1
ExpandDims3Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	2
Conv1D_1/conv1d/ExpandDims_1?
Conv1D_1/conv1dConv2D#Conv1D_1/conv1d/ExpandDims:output:0%Conv1D_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????	*
paddingVALID*
strides
2
Conv1D_1/conv1d?
Conv1D_1/conv1d/SqueezeSqueezeConv1D_1/conv1d:output:0*
T0*,
_output_shapes
:??????????	*
squeeze_dims

?????????2
Conv1D_1/conv1d/Squeeze?
Conv1D_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02!
Conv1D_1/BiasAdd/ReadVariableOp?
Conv1D_1/BiasAddBiasAdd Conv1D_1/conv1d/Squeeze:output:0'Conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????	2
Conv1D_1/BiasAddx
Conv1D_1/ReluReluConv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????	2
Conv1D_1/Relu?
MaxPooling1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_1/ExpandDims/dim?
MaxPooling1D_1/ExpandDims
ExpandDimsConv1D_1/Relu:activations:0&MaxPooling1D_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????	2
MaxPooling1D_1/ExpandDims?
MaxPooling1D_1/MaxPoolMaxPool"MaxPooling1D_1/ExpandDims:output:0*/
_output_shapes
:?????????J	*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_1/MaxPool?
MaxPooling1D_1/SqueezeSqueezeMaxPooling1D_1/MaxPool:output:0*
T0*+
_output_shapes
:?????????J	*
squeeze_dims
2
MaxPooling1D_1/Squeezes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *uf??2
dropout/dropout/Const?
dropout/dropout/MulMulMaxPooling1D_1/Squeeze:output:0dropout/dropout/Const:output:0*
T0*+
_output_shapes
:?????????J	2
dropout/dropout/Mul}
dropout/dropout/ShapeShapeMaxPooling1D_1/Squeeze:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????J	*
dtype0*
seed2????2.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *+?=2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????J	2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????J	2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????J	2
dropout/dropout/Mul_1?
Conv1D_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_2/conv1d/ExpandDims/dim?
Conv1D_2/conv1d/ExpandDims
ExpandDimsdropout/dropout/Mul_1:z:0'Conv1D_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????J	2
Conv1D_2/conv1d/ExpandDims?
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	*
dtype02-
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_2/conv1d/ExpandDims_1/dim?
Conv1D_2/conv1d/ExpandDims_1
ExpandDims3Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	2
Conv1D_2/conv1d/ExpandDims_1?
Conv1D_2/conv1dConv2D#Conv1D_2/conv1d/ExpandDims:output:0%Conv1D_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????G*
paddingVALID*
strides
2
Conv1D_2/conv1d?
Conv1D_2/conv1d/SqueezeSqueezeConv1D_2/conv1d:output:0*
T0*+
_output_shapes
:?????????G*
squeeze_dims

?????????2
Conv1D_2/conv1d/Squeeze?
Conv1D_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Conv1D_2/BiasAdd/ReadVariableOp?
Conv1D_2/BiasAddBiasAdd Conv1D_2/conv1d/Squeeze:output:0'Conv1D_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????G2
Conv1D_2/BiasAddw
Conv1D_2/ReluReluConv1D_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????G2
Conv1D_2/Relu?
MaxPooling1D_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_2/ExpandDims/dim?
MaxPooling1D_2/ExpandDims
ExpandDimsConv1D_2/Relu:activations:0&MaxPooling1D_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????G2
MaxPooling1D_2/ExpandDims?
MaxPooling1D_2/MaxPoolMaxPool"MaxPooling1D_2/ExpandDims:output:0*/
_output_shapes
:?????????#*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_2/MaxPool?
MaxPooling1D_2/SqueezeSqueezeMaxPooling1D_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????#*
squeeze_dims
2
MaxPooling1D_2/Squeezew
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *23??2
dropout_1/dropout/Const?
dropout_1/dropout/MulMulMaxPooling1D_2/Squeeze:output:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????#2
dropout_1/dropout/Mul?
dropout_1/dropout/ShapeShapeMaxPooling1D_2/Squeeze:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shape?
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????#*
dtype0*
seed220
.dropout_1/dropout/random_uniform/RandomUniform?
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>2"
 dropout_1/dropout/GreaterEqual/y?
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????#2 
dropout_1/dropout/GreaterEqual?
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????#2
dropout_1/dropout/Cast?
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????#2
dropout_1/dropout/Mul_1?
Conv1D_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_3/conv1d/ExpandDims/dim?
Conv1D_3/conv1d/ExpandDims
ExpandDimsdropout_1/dropout/Mul_1:z:0'Conv1D_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????#2
Conv1D_3/conv1d/ExpandDims?
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:.*
dtype02-
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_3/conv1d/ExpandDims_1/dim?
Conv1D_3/conv1d/ExpandDims_1
ExpandDims3Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:.2
Conv1D_3/conv1d/ExpandDims_1?
Conv1D_3/conv1dConv2D#Conv1D_3/conv1d/ExpandDims:output:0%Conv1D_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????.*
paddingVALID*
strides
2
Conv1D_3/conv1d?
Conv1D_3/conv1d/SqueezeSqueezeConv1D_3/conv1d:output:0*
T0*+
_output_shapes
:?????????.*
squeeze_dims

?????????2
Conv1D_3/conv1d/Squeeze?
Conv1D_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:.*
dtype02!
Conv1D_3/BiasAdd/ReadVariableOp?
Conv1D_3/BiasAddBiasAdd Conv1D_3/conv1d/Squeeze:output:0'Conv1D_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????.2
Conv1D_3/BiasAddw
Conv1D_3/ReluReluConv1D_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????.2
Conv1D_3/Relu?
MaxPooling1D_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_3/ExpandDims/dim?
MaxPooling1D_3/ExpandDims
ExpandDimsConv1D_3/Relu:activations:0&MaxPooling1D_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????.2
MaxPooling1D_3/ExpandDims?
MaxPooling1D_3/MaxPoolMaxPool"MaxPooling1D_3/ExpandDims:output:0*/
_output_shapes
:?????????.*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_3/MaxPool?
MaxPooling1D_3/SqueezeSqueezeMaxPooling1D_3/MaxPool:output:0*
T0*+
_output_shapes
:?????????.*
squeeze_dims
2
MaxPooling1D_3/Squeezew
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_2/dropout/Const?
dropout_2/dropout/MulMulMaxPooling1D_3/Squeeze:output:0 dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:?????????.2
dropout_2/dropout/Mul?
dropout_2/dropout/ShapeShapeMaxPooling1D_3/Squeeze:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shape?
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????.*
dtype0*
seed220
.dropout_2/dropout/random_uniform/RandomUniform?
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *^&?>2"
 dropout_2/dropout/GreaterEqual/y?
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????.2 
dropout_2/dropout/GreaterEqual?
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????.2
dropout_2/dropout/Cast?
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????.2
dropout_2/dropout/Mul_1?
Conv1D_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_4/conv1d/ExpandDims/dim?
Conv1D_4/conv1d/ExpandDims
ExpandDimsdropout_2/dropout/Mul_1:z:0'Conv1D_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????.2
Conv1D_4/conv1d/ExpandDims?
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:.*
dtype02-
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_4/conv1d/ExpandDims_1/dim?
Conv1D_4/conv1d/ExpandDims_1
ExpandDims3Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:.2
Conv1D_4/conv1d/ExpandDims_1?
Conv1D_4/conv1dConv2D#Conv1D_4/conv1d/ExpandDims:output:0%Conv1D_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv1D_4/conv1d?
Conv1D_4/conv1d/SqueezeSqueezeConv1D_4/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
Conv1D_4/conv1d/Squeeze?
Conv1D_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Conv1D_4/BiasAdd/ReadVariableOp?
Conv1D_4/BiasAddBiasAdd Conv1D_4/conv1d/Squeeze:output:0'Conv1D_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
Conv1D_4/BiasAddw
Conv1D_4/ReluReluConv1D_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
Conv1D_4/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????~   2
flatten/Const?
flatten/ReshapeReshapeConv1D_4/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????~2
flatten/Reshape?
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:~t*
dtype02
Dense_1/MatMul/ReadVariableOp?
Dense_1/MatMulMatMulflatten/Reshape:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t2
Dense_1/MatMul?
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:t*
dtype02 
Dense_1/BiasAdd/ReadVariableOp?
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t2
Dense_1/BiasAddp
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????t2
Dense_1/Relu?
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:t!*
dtype02
Dense_2/MatMul/ReadVariableOp?
Dense_2/MatMulMatMulDense_1/Relu:activations:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
Dense_2/MatMul?
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype02 
Dense_2/BiasAdd/ReadVariableOp?
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
Dense_2/BiasAddy
Dense_2/SoftmaxSoftmaxDense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????!2
Dense_2/Softmax?
IdentityIdentityDense_2/Softmax:softmax:0 ^Conv1D_1/BiasAdd/ReadVariableOp,^Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_2/BiasAdd/ReadVariableOp,^Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_3/BiasAdd/ReadVariableOp,^Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_4/BiasAdd/ReadVariableOp,^Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2B
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
:??????????
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_24936

inputsA
+conv1d_expanddims_1_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????J	2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????G*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????G*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????G2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????G2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????J	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????J	
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_24623
input_1
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:.
	unknown_4:.
	unknown_5:.
	unknown_6:
	unknown_7:~t
	unknown_8:t
	unknown_9:t!

unknown_10:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_240072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?
e
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_24046

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_24330

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *23??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????#2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????#*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????#2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????#2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????#2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????#:S O
+
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_25040

inputsA
+conv1d_expanddims_1_readvariableop_resource:.-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????.2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:.*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:.2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????.: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_24363

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *uf??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????J	2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????J	*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *+?=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????J	2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????J	2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????J	2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????J	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????J	:S O
+
_output_shapes
:?????????J	
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_24135

inputsA
+conv1d_expanddims_1_readvariableop_resource:.-
biasadd_readvariableop_resource:.
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????#2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:.*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:.2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????.*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????.*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:.*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????.2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????.2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????.2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
e
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_24031

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_25403
file_prefix6
 assignvariableop_conv1d_1_kernel:	.
 assignvariableop_1_conv1d_1_bias:	8
"assignvariableop_2_conv1d_2_kernel:	.
 assignvariableop_3_conv1d_2_bias:8
"assignvariableop_4_conv1d_3_kernel:..
 assignvariableop_5_conv1d_3_bias:.8
"assignvariableop_6_conv1d_4_kernel:..
 assignvariableop_7_conv1d_4_bias:3
!assignvariableop_8_dense_1_kernel:~t-
assignvariableop_9_dense_1_bias:t4
"assignvariableop_10_dense_2_kernel:t!.
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
*assignvariableop_21_adam_conv1d_1_kernel_m:	6
(assignvariableop_22_adam_conv1d_1_bias_m:	@
*assignvariableop_23_adam_conv1d_2_kernel_m:	6
(assignvariableop_24_adam_conv1d_2_bias_m:@
*assignvariableop_25_adam_conv1d_3_kernel_m:.6
(assignvariableop_26_adam_conv1d_3_bias_m:.@
*assignvariableop_27_adam_conv1d_4_kernel_m:.6
(assignvariableop_28_adam_conv1d_4_bias_m:;
)assignvariableop_29_adam_dense_1_kernel_m:~t5
'assignvariableop_30_adam_dense_1_bias_m:t;
)assignvariableop_31_adam_dense_2_kernel_m:t!5
'assignvariableop_32_adam_dense_2_bias_m:!@
*assignvariableop_33_adam_conv1d_1_kernel_v:	6
(assignvariableop_34_adam_conv1d_1_bias_v:	@
*assignvariableop_35_adam_conv1d_2_kernel_v:	6
(assignvariableop_36_adam_conv1d_2_bias_v:@
*assignvariableop_37_adam_conv1d_3_kernel_v:.6
(assignvariableop_38_adam_conv1d_3_bias_v:.@
*assignvariableop_39_adam_conv1d_4_kernel_v:.6
(assignvariableop_40_adam_conv1d_4_bias_v:;
)assignvariableop_41_adam_dense_1_kernel_v:~t5
'assignvariableop_42_adam_dense_1_bias_v:t;
)assignvariableop_43_adam_dense_2_kernel_v:t!5
'assignvariableop_44_adam_dense_2_bias_v:!
identity_46??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*?
value?B?.B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:.*
dtype0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::*<
dtypes2
02.	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv1d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv1d_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv1d_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv1d_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv1d_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv1d_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv1d_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv1d_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv1d_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv1d_3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv1d_3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv1d_4_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv1d_4_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv1d_1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv1d_1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv1d_2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv1d_2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv1d_3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv1d_3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv1d_4_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv1d_4_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_449
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_45Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_45?
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
?
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_24962

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *23??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????#2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????#*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????#2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????#2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????#2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????#:S O
+
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_24297

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????.2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????.*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *^&?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????.2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????.2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????.2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????.2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????.:S O
+
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_24950

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????#2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????#2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????#:S O
+
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
`
'__inference_dropout_layer_call_fn_24920

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????J	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_243632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????J	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????J	22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????J	
 
_user_specified_nameinputs
?
?
(__inference_Conv1D_4_layer_call_fn_25049

inputs
unknown:.
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_241652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????.: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_24177

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????~   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????~2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????~2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?q
?

@__inference_model_layer_call_and_return_conditional_losses_24706

inputsJ
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:	6
(conv1d_1_biasadd_readvariableop_resource:	J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:	6
(conv1d_2_biasadd_readvariableop_resource:J
4conv1d_3_conv1d_expanddims_1_readvariableop_resource:.6
(conv1d_3_biasadd_readvariableop_resource:.J
4conv1d_4_conv1d_expanddims_1_readvariableop_resource:.6
(conv1d_4_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:~t5
'dense_1_biasadd_readvariableop_resource:t8
&dense_2_matmul_readvariableop_resource:t!5
'dense_2_biasadd_readvariableop_resource:!
identity??Conv1D_1/BiasAdd/ReadVariableOp?+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_2/BiasAdd/ReadVariableOp?+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_3/BiasAdd/ReadVariableOp?+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?Conv1D_4/BiasAdd/ReadVariableOp?+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?Dense_1/BiasAdd/ReadVariableOp?Dense_1/MatMul/ReadVariableOp?Dense_2/BiasAdd/ReadVariableOp?Dense_2/MatMul/ReadVariableOp?
Conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_1/conv1d/ExpandDims/dim?
Conv1D_1/conv1d/ExpandDims
ExpandDimsinputs'Conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
Conv1D_1/conv1d/ExpandDims?
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	*
dtype02-
+Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_1/conv1d/ExpandDims_1/dim?
Conv1D_1/conv1d/ExpandDims_1
ExpandDims3Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	2
Conv1D_1/conv1d/ExpandDims_1?
Conv1D_1/conv1dConv2D#Conv1D_1/conv1d/ExpandDims:output:0%Conv1D_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????	*
paddingVALID*
strides
2
Conv1D_1/conv1d?
Conv1D_1/conv1d/SqueezeSqueezeConv1D_1/conv1d:output:0*
T0*,
_output_shapes
:??????????	*
squeeze_dims

?????????2
Conv1D_1/conv1d/Squeeze?
Conv1D_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02!
Conv1D_1/BiasAdd/ReadVariableOp?
Conv1D_1/BiasAddBiasAdd Conv1D_1/conv1d/Squeeze:output:0'Conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????	2
Conv1D_1/BiasAddx
Conv1D_1/ReluReluConv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????	2
Conv1D_1/Relu?
MaxPooling1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_1/ExpandDims/dim?
MaxPooling1D_1/ExpandDims
ExpandDimsConv1D_1/Relu:activations:0&MaxPooling1D_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????	2
MaxPooling1D_1/ExpandDims?
MaxPooling1D_1/MaxPoolMaxPool"MaxPooling1D_1/ExpandDims:output:0*/
_output_shapes
:?????????J	*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_1/MaxPool?
MaxPooling1D_1/SqueezeSqueezeMaxPooling1D_1/MaxPool:output:0*
T0*+
_output_shapes
:?????????J	*
squeeze_dims
2
MaxPooling1D_1/Squeeze?
dropout/IdentityIdentityMaxPooling1D_1/Squeeze:output:0*
T0*+
_output_shapes
:?????????J	2
dropout/Identity?
Conv1D_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_2/conv1d/ExpandDims/dim?
Conv1D_2/conv1d/ExpandDims
ExpandDimsdropout/Identity:output:0'Conv1D_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????J	2
Conv1D_2/conv1d/ExpandDims?
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	*
dtype02-
+Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_2/conv1d/ExpandDims_1/dim?
Conv1D_2/conv1d/ExpandDims_1
ExpandDims3Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	2
Conv1D_2/conv1d/ExpandDims_1?
Conv1D_2/conv1dConv2D#Conv1D_2/conv1d/ExpandDims:output:0%Conv1D_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????G*
paddingVALID*
strides
2
Conv1D_2/conv1d?
Conv1D_2/conv1d/SqueezeSqueezeConv1D_2/conv1d:output:0*
T0*+
_output_shapes
:?????????G*
squeeze_dims

?????????2
Conv1D_2/conv1d/Squeeze?
Conv1D_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Conv1D_2/BiasAdd/ReadVariableOp?
Conv1D_2/BiasAddBiasAdd Conv1D_2/conv1d/Squeeze:output:0'Conv1D_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????G2
Conv1D_2/BiasAddw
Conv1D_2/ReluReluConv1D_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????G2
Conv1D_2/Relu?
MaxPooling1D_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_2/ExpandDims/dim?
MaxPooling1D_2/ExpandDims
ExpandDimsConv1D_2/Relu:activations:0&MaxPooling1D_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????G2
MaxPooling1D_2/ExpandDims?
MaxPooling1D_2/MaxPoolMaxPool"MaxPooling1D_2/ExpandDims:output:0*/
_output_shapes
:?????????#*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_2/MaxPool?
MaxPooling1D_2/SqueezeSqueezeMaxPooling1D_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????#*
squeeze_dims
2
MaxPooling1D_2/Squeeze?
dropout_1/IdentityIdentityMaxPooling1D_2/Squeeze:output:0*
T0*+
_output_shapes
:?????????#2
dropout_1/Identity?
Conv1D_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_3/conv1d/ExpandDims/dim?
Conv1D_3/conv1d/ExpandDims
ExpandDimsdropout_1/Identity:output:0'Conv1D_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????#2
Conv1D_3/conv1d/ExpandDims?
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:.*
dtype02-
+Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_3/conv1d/ExpandDims_1/dim?
Conv1D_3/conv1d/ExpandDims_1
ExpandDims3Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:.2
Conv1D_3/conv1d/ExpandDims_1?
Conv1D_3/conv1dConv2D#Conv1D_3/conv1d/ExpandDims:output:0%Conv1D_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????.*
paddingVALID*
strides
2
Conv1D_3/conv1d?
Conv1D_3/conv1d/SqueezeSqueezeConv1D_3/conv1d:output:0*
T0*+
_output_shapes
:?????????.*
squeeze_dims

?????????2
Conv1D_3/conv1d/Squeeze?
Conv1D_3/BiasAdd/ReadVariableOpReadVariableOp(conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:.*
dtype02!
Conv1D_3/BiasAdd/ReadVariableOp?
Conv1D_3/BiasAddBiasAdd Conv1D_3/conv1d/Squeeze:output:0'Conv1D_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????.2
Conv1D_3/BiasAddw
Conv1D_3/ReluReluConv1D_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????.2
Conv1D_3/Relu?
MaxPooling1D_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
MaxPooling1D_3/ExpandDims/dim?
MaxPooling1D_3/ExpandDims
ExpandDimsConv1D_3/Relu:activations:0&MaxPooling1D_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????.2
MaxPooling1D_3/ExpandDims?
MaxPooling1D_3/MaxPoolMaxPool"MaxPooling1D_3/ExpandDims:output:0*/
_output_shapes
:?????????.*
ksize
*
paddingVALID*
strides
2
MaxPooling1D_3/MaxPool?
MaxPooling1D_3/SqueezeSqueezeMaxPooling1D_3/MaxPool:output:0*
T0*+
_output_shapes
:?????????.*
squeeze_dims
2
MaxPooling1D_3/Squeeze?
dropout_2/IdentityIdentityMaxPooling1D_3/Squeeze:output:0*
T0*+
_output_shapes
:?????????.2
dropout_2/Identity?
Conv1D_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
Conv1D_4/conv1d/ExpandDims/dim?
Conv1D_4/conv1d/ExpandDims
ExpandDimsdropout_2/Identity:output:0'Conv1D_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????.2
Conv1D_4/conv1d/ExpandDims?
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:.*
dtype02-
+Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?
 Conv1D_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 Conv1D_4/conv1d/ExpandDims_1/dim?
Conv1D_4/conv1d/ExpandDims_1
ExpandDims3Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp:value:0)Conv1D_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:.2
Conv1D_4/conv1d/ExpandDims_1?
Conv1D_4/conv1dConv2D#Conv1D_4/conv1d/ExpandDims:output:0%Conv1D_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv1D_4/conv1d?
Conv1D_4/conv1d/SqueezeSqueezeConv1D_4/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
Conv1D_4/conv1d/Squeeze?
Conv1D_4/BiasAdd/ReadVariableOpReadVariableOp(conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
Conv1D_4/BiasAdd/ReadVariableOp?
Conv1D_4/BiasAddBiasAdd Conv1D_4/conv1d/Squeeze:output:0'Conv1D_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
Conv1D_4/BiasAddw
Conv1D_4/ReluReluConv1D_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
Conv1D_4/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????~   2
flatten/Const?
flatten/ReshapeReshapeConv1D_4/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:?????????~2
flatten/Reshape?
Dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:~t*
dtype02
Dense_1/MatMul/ReadVariableOp?
Dense_1/MatMulMatMulflatten/Reshape:output:0%Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t2
Dense_1/MatMul?
Dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:t*
dtype02 
Dense_1/BiasAdd/ReadVariableOp?
Dense_1/BiasAddBiasAddDense_1/MatMul:product:0&Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t2
Dense_1/BiasAddp
Dense_1/ReluReluDense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????t2
Dense_1/Relu?
Dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:t!*
dtype02
Dense_2/MatMul/ReadVariableOp?
Dense_2/MatMulMatMulDense_1/Relu:activations:0%Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
Dense_2/MatMul?
Dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype02 
Dense_2/BiasAdd/ReadVariableOp?
Dense_2/BiasAddBiasAddDense_2/MatMul:product:0&Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
Dense_2/BiasAddy
Dense_2/SoftmaxSoftmaxDense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????!2
Dense_2/Softmax?
IdentityIdentityDense_2/Softmax:softmax:0 ^Conv1D_1/BiasAdd/ReadVariableOp,^Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_2/BiasAdd/ReadVariableOp,^Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_3/BiasAdd/ReadVariableOp,^Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp ^Conv1D_4/BiasAdd/ReadVariableOp,^Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp^Dense_1/BiasAdd/ReadVariableOp^Dense_1/MatMul/ReadVariableOp^Dense_2/BiasAdd/ReadVariableOp^Dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2B
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
:??????????
 
_user_specified_nameinputs
?
C
'__inference_dropout_layer_call_fn_24915

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????J	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_240872
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????J	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????J	:S O
+
_output_shapes
:?????????J	
 
_user_specified_nameinputs
?
?
'__inference_Dense_2_layer_call_fn_25100

inputs
unknown:t!
	unknown_0:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Dense_2_layer_call_and_return_conditional_losses_242072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????t: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????t
 
_user_specified_nameinputs
?
e
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_24016

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
c
D__inference_dropout_2_layer_call_and_return_conditional_losses_25014

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????.2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????.*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *^&?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????.2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????.2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????.2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????.2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????.:S O
+
_output_shapes
:?????????.
 
_user_specified_nameinputs
?
J
.__inference_MaxPooling1D_2_layer_call_fn_24037

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_240312
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?

?
B__inference_Dense_1_layer_call_and_return_conditional_losses_24190

inputs0
matmul_readvariableop_resource:~t-
biasadd_readvariableop_resource:t
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:~t*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:t*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????t2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????t2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????~: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????~
 
_user_specified_nameinputs
?
J
.__inference_MaxPooling1D_1_layer_call_fn_24022

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_240162
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_24898

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????J	2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????J	2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????J	:S O
+
_output_shapes
:?????????J	
 
_user_specified_nameinputs
?
?
(__inference_Conv1D_2_layer_call_fn_24945

inputs
unknown:	
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????G*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_241052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????J	: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????J	
 
_user_specified_nameinputs
?
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_24117

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????#2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????#2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????#:S O
+
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_24105

inputsA
+conv1d_expanddims_1_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????J	2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????G*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????G*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????G2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????G2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????G2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????J	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????J	
 
_user_specified_nameinputs
?

?
B__inference_Dense_1_layer_call_and_return_conditional_losses_25071

inputs0
matmul_readvariableop_resource:~t-
biasadd_readvariableop_resource:t
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:~t*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:t*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????t2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????t2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????~: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????~
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_24075

inputsA
+conv1d_expanddims_1_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????	*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????	*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????	2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????	2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_24504
input_1
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:.
	unknown_4:.
	unknown_5:.
	unknown_6:
	unknown_7:~t
	unknown_8:t
	unknown_9:t!

unknown_10:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_244482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
(__inference_Conv1D_3_layer_call_fn_24997

inputs
unknown:.
	unknown_0:.
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????.*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_241352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????.2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????#: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_25002

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????.2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????.2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????.:S O
+
_output_shapes
:?????????.
 
_user_specified_nameinputs
?

?
%__inference_model_layer_call_fn_24868

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:
	unknown_3:.
	unknown_4:.
	unknown_5:.
	unknown_6:
	unknown_7:~t
	unknown_8:t
	unknown_9:t!

unknown_10:!
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????!*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_244482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_24988

inputsA
+conv1d_expanddims_1_readvariableop_resource:.-
biasadd_readvariableop_resource:.
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????#2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:.*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:.2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????.*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????.*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:.*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????.2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????.2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????.2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????#
 
_user_specified_nameinputs
?}
?
 __inference__wrapped_model_24007
input_1P
:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource:	<
.model_conv1d_1_biasadd_readvariableop_resource:	P
:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource:	<
.model_conv1d_2_biasadd_readvariableop_resource:P
:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource:.<
.model_conv1d_3_biasadd_readvariableop_resource:.P
:model_conv1d_4_conv1d_expanddims_1_readvariableop_resource:.<
.model_conv1d_4_biasadd_readvariableop_resource:>
,model_dense_1_matmul_readvariableop_resource:~t;
-model_dense_1_biasadd_readvariableop_resource:t>
,model_dense_2_matmul_readvariableop_resource:t!;
-model_dense_2_biasadd_readvariableop_resource:!
identity??%model/Conv1D_1/BiasAdd/ReadVariableOp?1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?%model/Conv1D_2/BiasAdd/ReadVariableOp?1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?%model/Conv1D_3/BiasAdd/ReadVariableOp?1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?%model/Conv1D_4/BiasAdd/ReadVariableOp?1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?$model/Dense_1/BiasAdd/ReadVariableOp?#model/Dense_1/MatMul/ReadVariableOp?$model/Dense_2/BiasAdd/ReadVariableOp?#model/Dense_2/MatMul/ReadVariableOp?
$model/Conv1D_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/Conv1D_1/conv1d/ExpandDims/dim?
 model/Conv1D_1/conv1d/ExpandDims
ExpandDimsinput_1-model/Conv1D_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2"
 model/Conv1D_1/conv1d/ExpandDims?
1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	*
dtype023
1model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp?
&model/Conv1D_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_1/conv1d/ExpandDims_1/dim?
"model/Conv1D_1/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	2$
"model/Conv1D_1/conv1d/ExpandDims_1?
model/Conv1D_1/conv1dConv2D)model/Conv1D_1/conv1d/ExpandDims:output:0+model/Conv1D_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????	*
paddingVALID*
strides
2
model/Conv1D_1/conv1d?
model/Conv1D_1/conv1d/SqueezeSqueezemodel/Conv1D_1/conv1d:output:0*
T0*,
_output_shapes
:??????????	*
squeeze_dims

?????????2
model/Conv1D_1/conv1d/Squeeze?
%model/Conv1D_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02'
%model/Conv1D_1/BiasAdd/ReadVariableOp?
model/Conv1D_1/BiasAddBiasAdd&model/Conv1D_1/conv1d/Squeeze:output:0-model/Conv1D_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????	2
model/Conv1D_1/BiasAdd?
model/Conv1D_1/ReluRelumodel/Conv1D_1/BiasAdd:output:0*
T0*,
_output_shapes
:??????????	2
model/Conv1D_1/Relu?
#model/MaxPooling1D_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/MaxPooling1D_1/ExpandDims/dim?
model/MaxPooling1D_1/ExpandDims
ExpandDims!model/Conv1D_1/Relu:activations:0,model/MaxPooling1D_1/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????	2!
model/MaxPooling1D_1/ExpandDims?
model/MaxPooling1D_1/MaxPoolMaxPool(model/MaxPooling1D_1/ExpandDims:output:0*/
_output_shapes
:?????????J	*
ksize
*
paddingVALID*
strides
2
model/MaxPooling1D_1/MaxPool?
model/MaxPooling1D_1/SqueezeSqueeze%model/MaxPooling1D_1/MaxPool:output:0*
T0*+
_output_shapes
:?????????J	*
squeeze_dims
2
model/MaxPooling1D_1/Squeeze?
model/dropout/IdentityIdentity%model/MaxPooling1D_1/Squeeze:output:0*
T0*+
_output_shapes
:?????????J	2
model/dropout/Identity?
$model/Conv1D_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/Conv1D_2/conv1d/ExpandDims/dim?
 model/Conv1D_2/conv1d/ExpandDims
ExpandDimsmodel/dropout/Identity:output:0-model/Conv1D_2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????J	2"
 model/Conv1D_2/conv1d/ExpandDims?
1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	*
dtype023
1model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp?
&model/Conv1D_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_2/conv1d/ExpandDims_1/dim?
"model/Conv1D_2/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	2$
"model/Conv1D_2/conv1d/ExpandDims_1?
model/Conv1D_2/conv1dConv2D)model/Conv1D_2/conv1d/ExpandDims:output:0+model/Conv1D_2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????G*
paddingVALID*
strides
2
model/Conv1D_2/conv1d?
model/Conv1D_2/conv1d/SqueezeSqueezemodel/Conv1D_2/conv1d:output:0*
T0*+
_output_shapes
:?????????G*
squeeze_dims

?????????2
model/Conv1D_2/conv1d/Squeeze?
%model/Conv1D_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/Conv1D_2/BiasAdd/ReadVariableOp?
model/Conv1D_2/BiasAddBiasAdd&model/Conv1D_2/conv1d/Squeeze:output:0-model/Conv1D_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????G2
model/Conv1D_2/BiasAdd?
model/Conv1D_2/ReluRelumodel/Conv1D_2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????G2
model/Conv1D_2/Relu?
#model/MaxPooling1D_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/MaxPooling1D_2/ExpandDims/dim?
model/MaxPooling1D_2/ExpandDims
ExpandDims!model/Conv1D_2/Relu:activations:0,model/MaxPooling1D_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????G2!
model/MaxPooling1D_2/ExpandDims?
model/MaxPooling1D_2/MaxPoolMaxPool(model/MaxPooling1D_2/ExpandDims:output:0*/
_output_shapes
:?????????#*
ksize
*
paddingVALID*
strides
2
model/MaxPooling1D_2/MaxPool?
model/MaxPooling1D_2/SqueezeSqueeze%model/MaxPooling1D_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????#*
squeeze_dims
2
model/MaxPooling1D_2/Squeeze?
model/dropout_1/IdentityIdentity%model/MaxPooling1D_2/Squeeze:output:0*
T0*+
_output_shapes
:?????????#2
model/dropout_1/Identity?
$model/Conv1D_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/Conv1D_3/conv1d/ExpandDims/dim?
 model/Conv1D_3/conv1d/ExpandDims
ExpandDims!model/dropout_1/Identity:output:0-model/Conv1D_3/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????#2"
 model/Conv1D_3/conv1d/ExpandDims?
1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:.*
dtype023
1model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp?
&model/Conv1D_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_3/conv1d/ExpandDims_1/dim?
"model/Conv1D_3/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:.2$
"model/Conv1D_3/conv1d/ExpandDims_1?
model/Conv1D_3/conv1dConv2D)model/Conv1D_3/conv1d/ExpandDims:output:0+model/Conv1D_3/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????.*
paddingVALID*
strides
2
model/Conv1D_3/conv1d?
model/Conv1D_3/conv1d/SqueezeSqueezemodel/Conv1D_3/conv1d:output:0*
T0*+
_output_shapes
:?????????.*
squeeze_dims

?????????2
model/Conv1D_3/conv1d/Squeeze?
%model/Conv1D_3/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:.*
dtype02'
%model/Conv1D_3/BiasAdd/ReadVariableOp?
model/Conv1D_3/BiasAddBiasAdd&model/Conv1D_3/conv1d/Squeeze:output:0-model/Conv1D_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????.2
model/Conv1D_3/BiasAdd?
model/Conv1D_3/ReluRelumodel/Conv1D_3/BiasAdd:output:0*
T0*+
_output_shapes
:?????????.2
model/Conv1D_3/Relu?
#model/MaxPooling1D_3/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#model/MaxPooling1D_3/ExpandDims/dim?
model/MaxPooling1D_3/ExpandDims
ExpandDims!model/Conv1D_3/Relu:activations:0,model/MaxPooling1D_3/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????.2!
model/MaxPooling1D_3/ExpandDims?
model/MaxPooling1D_3/MaxPoolMaxPool(model/MaxPooling1D_3/ExpandDims:output:0*/
_output_shapes
:?????????.*
ksize
*
paddingVALID*
strides
2
model/MaxPooling1D_3/MaxPool?
model/MaxPooling1D_3/SqueezeSqueeze%model/MaxPooling1D_3/MaxPool:output:0*
T0*+
_output_shapes
:?????????.*
squeeze_dims
2
model/MaxPooling1D_3/Squeeze?
model/dropout_2/IdentityIdentity%model/MaxPooling1D_3/Squeeze:output:0*
T0*+
_output_shapes
:?????????.2
model/dropout_2/Identity?
$model/Conv1D_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$model/Conv1D_4/conv1d/ExpandDims/dim?
 model/Conv1D_4/conv1d/ExpandDims
ExpandDims!model/dropout_2/Identity:output:0-model/Conv1D_4/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????.2"
 model/Conv1D_4/conv1d/ExpandDims?
1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:model_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:.*
dtype023
1model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp?
&model/Conv1D_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&model/Conv1D_4/conv1d/ExpandDims_1/dim?
"model/Conv1D_4/conv1d/ExpandDims_1
ExpandDims9model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp:value:0/model/Conv1D_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:.2$
"model/Conv1D_4/conv1d/ExpandDims_1?
model/Conv1D_4/conv1dConv2D)model/Conv1D_4/conv1d/ExpandDims:output:0+model/Conv1D_4/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model/Conv1D_4/conv1d?
model/Conv1D_4/conv1d/SqueezeSqueezemodel/Conv1D_4/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
model/Conv1D_4/conv1d/Squeeze?
%model/Conv1D_4/BiasAdd/ReadVariableOpReadVariableOp.model_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%model/Conv1D_4/BiasAdd/ReadVariableOp?
model/Conv1D_4/BiasAddBiasAdd&model/Conv1D_4/conv1d/Squeeze:output:0-model/Conv1D_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
model/Conv1D_4/BiasAdd?
model/Conv1D_4/ReluRelumodel/Conv1D_4/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
model/Conv1D_4/Relu{
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????~   2
model/flatten/Const?
model/flatten/ReshapeReshape!model/Conv1D_4/Relu:activations:0model/flatten/Const:output:0*
T0*'
_output_shapes
:?????????~2
model/flatten/Reshape?
#model/Dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:~t*
dtype02%
#model/Dense_1/MatMul/ReadVariableOp?
model/Dense_1/MatMulMatMulmodel/flatten/Reshape:output:0+model/Dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t2
model/Dense_1/MatMul?
$model/Dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:t*
dtype02&
$model/Dense_1/BiasAdd/ReadVariableOp?
model/Dense_1/BiasAddBiasAddmodel/Dense_1/MatMul:product:0,model/Dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t2
model/Dense_1/BiasAdd?
model/Dense_1/ReluRelumodel/Dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????t2
model/Dense_1/Relu?
#model/Dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

:t!*
dtype02%
#model/Dense_2/MatMul/ReadVariableOp?
model/Dense_2/MatMulMatMul model/Dense_1/Relu:activations:0+model/Dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
model/Dense_2/MatMul?
$model/Dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:!*
dtype02&
$model/Dense_2/BiasAdd/ReadVariableOp?
model/Dense_2/BiasAddBiasAddmodel/Dense_2/MatMul:product:0,model/Dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
model/Dense_2/BiasAdd?
model/Dense_2/SoftmaxSoftmaxmodel/Dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????!2
model/Dense_2/Softmax?
IdentityIdentitymodel/Dense_2/Softmax:softmax:0&^model/Conv1D_1/BiasAdd/ReadVariableOp2^model/Conv1D_1/conv1d/ExpandDims_1/ReadVariableOp&^model/Conv1D_2/BiasAdd/ReadVariableOp2^model/Conv1D_2/conv1d/ExpandDims_1/ReadVariableOp&^model/Conv1D_3/BiasAdd/ReadVariableOp2^model/Conv1D_3/conv1d/ExpandDims_1/ReadVariableOp&^model/Conv1D_4/BiasAdd/ReadVariableOp2^model/Conv1D_4/conv1d/ExpandDims_1/ReadVariableOp%^model/Dense_1/BiasAdd/ReadVariableOp$^model/Dense_1/MatMul/ReadVariableOp%^model/Dense_2/BiasAdd/ReadVariableOp$^model/Dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:??????????: : : : : : : : : : : : 2N
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
:??????????
!
_user_specified_name	input_1
?
a
B__inference_dropout_layer_call_and_return_conditional_losses_24910

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *uf??2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????J	2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????J	*
dtype0*
seed2????2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *+?=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????J	2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????J	2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????J	2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????J	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????J	:S O
+
_output_shapes
:?????????J	
 
_user_specified_nameinputs
?
E
)__inference_dropout_2_layer_call_fn_25019

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????.* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_241472
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????.2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????.:S O
+
_output_shapes
:?????????.
 
_user_specified_nameinputs
?

?
B__inference_Dense_2_layer_call_and_return_conditional_losses_25091

inputs0
matmul_readvariableop_resource:t!-
biasadd_readvariableop_resource:!
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:t!*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:!*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????!2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????!2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????!2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????t: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????t
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_24884

inputsA
+conv1d_expanddims_1_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:??????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:	*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:	2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:??????????	*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:??????????	*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????	2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????	2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
B__inference_dropout_layer_call_and_return_conditional_losses_24087

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????J	2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????J	2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????J	:S O
+
_output_shapes
:?????????J	
 
_user_specified_nameinputs
?
b
)__inference_dropout_1_layer_call_fn_24972

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_243302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????#22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
?
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_24165

inputsA
+conv1d_expanddims_1_readvariableop_resource:.-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????.2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:.*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:.2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????.: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????.
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input_15
serving_default_input_1:0??????????;
Dense_20
StatefulPartitionedCall:0?????????!tensorflow/serving/predict:??
?t
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer_with_weights-4
layer-12
layer_with_weights-5
layer-13
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?o
_tf_keras_network?o{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "Conv1D_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_1", "inbound_nodes": [[["Conv1D_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.11967118727656578, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["MaxPooling1D_1", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_2", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_2", "inbound_nodes": [[["Conv1D_2", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.334026999610696, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["MaxPooling1D_2", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3", "trainable": true, "dtype": "float32", "filters": 46, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3", "inbound_nodes": [[["dropout_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_3", "inbound_nodes": [[["Conv1D_3", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25029273205218944, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["MaxPooling1D_3", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "Conv1D_4", "trainable": true, "dtype": "float32", "filters": 21, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_4", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["Conv1D_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 116, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 33, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_2", "inbound_nodes": [[["Dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Dense_2", 0, 0]]}, "shared_object_id": 26, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 300, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "Conv1D_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_1", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_1", "inbound_nodes": [[["Conv1D_1", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.11967118727656578, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["MaxPooling1D_1", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Conv1D", "config": {"name": "Conv1D_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_2", "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_2", "inbound_nodes": [[["Conv1D_2", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.334026999610696, "noise_shape": null, "seed": null}, "name": "dropout_1", "inbound_nodes": [[["MaxPooling1D_2", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Conv1D", "config": {"name": "Conv1D_3", "trainable": true, "dtype": "float32", "filters": 46, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_3", "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "name": "MaxPooling1D_3", "inbound_nodes": [[["Conv1D_3", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25029273205218944, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["MaxPooling1D_3", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Conv1D", "config": {"name": "Conv1D_4", "trainable": true, "dtype": "float32", "filters": 21, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Conv1D_4", "inbound_nodes": [[["dropout_2", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["Conv1D_4", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 116, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_1", "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 33, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Dense_2", "inbound_nodes": [[["Dense_1", 0, 0, {}]]], "shared_object_id": 25}], "input_layers": [["input_1", 0, 0]], "output_layers": [["Dense_2", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 28}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 3.9999998989515007e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?
{"name": "Conv1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 300, 1]}, "dtype": "float32", "filters": 9, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}, "shared_object_id": 29}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 300, 1]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "MaxPooling1D_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_1", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_1", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 30}}
?
trainable_variables
 regularization_losses
!	variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.11967118727656578, "noise_shape": null, "seed": null}, "inbound_nodes": [[["MaxPooling1D_1", 0, 0, {}]]], "shared_object_id": 5}
?


#kernel
$bias
%trainable_variables
&regularization_losses
'	variables
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "Conv1D_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 9}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 74, 9]}}
?
)trainable_variables
*regularization_losses
+	variables
,	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "MaxPooling1D_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_2", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 32}}
?
-trainable_variables
.regularization_losses
/	variables
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.334026999610696, "noise_shape": null, "seed": null}, "inbound_nodes": [[["MaxPooling1D_2", 0, 0, {}]]], "shared_object_id": 10}
?


1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "Conv1D_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_3", "trainable": true, "dtype": "float32", "filters": 46, "kernel_size": {"class_name": "__tuple__", "items": [6]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_1", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 16}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35, 16]}}
?
7trainable_variables
8regularization_losses
9	variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "MaxPooling1D_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling1D", "config": {"name": "MaxPooling1D_3", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [4]}, "pool_size": {"class_name": "__tuple__", "items": [4]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_3", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 34}}
?
;trainable_variables
<regularization_losses
=	variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.25029273205218944, "noise_shape": null, "seed": null}, "inbound_nodes": [[["MaxPooling1D_3", 0, 0, {}]]], "shared_object_id": 15}
?


?kernel
@bias
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "Conv1D_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv1D", "config": {"name": "Conv1D_4", "trainable": true, "dtype": "float32", "filters": 21, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_2", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 46}}, "shared_object_id": 35}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 46]}}
?
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["Conv1D_4", 0, 0, {}]]], "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 36}}
?	

Ikernel
Jbias
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "Dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Dense_1", "trainable": true, "dtype": "float32", "units": 116, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 126}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 126]}}
?	

Okernel
Pbias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "Dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "Dense_2", "trainable": true, "dtype": "float32", "units": 33, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["Dense_1", 0, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 116}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 116]}}
?
Uiter

Vbeta_1

Wbeta_2
	Xdecay
Ylearning_ratem?m?#m?$m?1m?2m??m?@m?Im?Jm?Om?Pm?v?v?#v?$v?1v?2v??v?@v?Iv?Jv?Ov?Pv?"
	optimizer
 "
trackable_list_wrapper
v
0
1
#2
$3
14
25
?6
@7
I8
J9
O10
P11"
trackable_list_wrapper
v
0
1
#2
$3
14
25
?6
@7
I8
J9
O10
P11"
trackable_list_wrapper
?
Zmetrics
regularization_losses

[layers
\non_trainable_variables
trainable_variables
]layer_metrics
	variables
^layer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
%:#	2Conv1D_1/kernel
:	2Conv1D_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
_metrics

`layers
anon_trainable_variables
trainable_variables
regularization_losses
blayer_metrics
	variables
clayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
dmetrics

elayers
fnon_trainable_variables
trainable_variables
regularization_losses
glayer_metrics
	variables
hlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
imetrics

jlayers
knon_trainable_variables
trainable_variables
 regularization_losses
llayer_metrics
!	variables
mlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#	2Conv1D_2/kernel
:2Conv1D_2/bias
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
nmetrics

olayers
pnon_trainable_variables
%trainable_variables
&regularization_losses
qlayer_metrics
'	variables
rlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
smetrics

tlayers
unon_trainable_variables
)trainable_variables
*regularization_losses
vlayer_metrics
+	variables
wlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
xmetrics

ylayers
znon_trainable_variables
-trainable_variables
.regularization_losses
{layer_metrics
/	variables
|layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#.2Conv1D_3/kernel
:.2Conv1D_3/bias
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
}metrics

~layers
non_trainable_variables
3trainable_variables
4regularization_losses
?layer_metrics
5	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
?non_trainable_variables
7trainable_variables
8regularization_losses
?layer_metrics
9	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
?non_trainable_variables
;trainable_variables
<regularization_losses
?layer_metrics
=	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#.2Conv1D_4/kernel
:2Conv1D_4/bias
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
?
?metrics
?layers
?non_trainable_variables
Atrainable_variables
Bregularization_losses
?layer_metrics
C	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
?non_trainable_variables
Etrainable_variables
Fregularization_losses
?layer_metrics
G	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :~t2Dense_1/kernel
:t2Dense_1/bias
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
?
?metrics
?layers
?non_trainable_variables
Ktrainable_variables
Lregularization_losses
?layer_metrics
M	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :t!2Dense_2/kernel
:!2Dense_2/bias
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
?
?metrics
?layers
?non_trainable_variables
Qtrainable_variables
Rregularization_losses
?layer_metrics
S	variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0
?0
?1"
trackable_list_wrapper
?
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
13"
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
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 39}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 28}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
*:(	2Adam/Conv1D_1/kernel/m
 :	2Adam/Conv1D_1/bias/m
*:(	2Adam/Conv1D_2/kernel/m
 :2Adam/Conv1D_2/bias/m
*:(.2Adam/Conv1D_3/kernel/m
 :.2Adam/Conv1D_3/bias/m
*:(.2Adam/Conv1D_4/kernel/m
 :2Adam/Conv1D_4/bias/m
%:#~t2Adam/Dense_1/kernel/m
:t2Adam/Dense_1/bias/m
%:#t!2Adam/Dense_2/kernel/m
:!2Adam/Dense_2/bias/m
*:(	2Adam/Conv1D_1/kernel/v
 :	2Adam/Conv1D_1/bias/v
*:(	2Adam/Conv1D_2/kernel/v
 :2Adam/Conv1D_2/bias/v
*:(.2Adam/Conv1D_3/kernel/v
 :.2Adam/Conv1D_3/bias/v
*:(.2Adam/Conv1D_4/kernel/v
 :2Adam/Conv1D_4/bias/v
%:#~t2Adam/Dense_1/kernel/v
:t2Adam/Dense_1/bias/v
%:#t!2Adam/Dense_2/kernel/v
:!2Adam/Dense_2/bias/v
?2?
 __inference__wrapped_model_24007?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *+?(
&?#
input_1??????????
?2?
@__inference_model_layer_call_and_return_conditional_losses_24706
@__inference_model_layer_call_and_return_conditional_losses_24810
@__inference_model_layer_call_and_return_conditional_losses_24545
@__inference_model_layer_call_and_return_conditional_losses_24586?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_model_layer_call_fn_24241
%__inference_model_layer_call_fn_24839
%__inference_model_layer_call_fn_24868
%__inference_model_layer_call_fn_24504?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_24884?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_Conv1D_1_layer_call_fn_24893?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_24016?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
.__inference_MaxPooling1D_1_layer_call_fn_24022?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
B__inference_dropout_layer_call_and_return_conditional_losses_24898
B__inference_dropout_layer_call_and_return_conditional_losses_24910?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_dropout_layer_call_fn_24915
'__inference_dropout_layer_call_fn_24920?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_24936?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_Conv1D_2_layer_call_fn_24945?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_24031?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
.__inference_MaxPooling1D_2_layer_call_fn_24037?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
D__inference_dropout_1_layer_call_and_return_conditional_losses_24950
D__inference_dropout_1_layer_call_and_return_conditional_losses_24962?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_1_layer_call_fn_24967
)__inference_dropout_1_layer_call_fn_24972?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_24988?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_Conv1D_3_layer_call_fn_24997?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_24046?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
.__inference_MaxPooling1D_3_layer_call_fn_24052?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
D__inference_dropout_2_layer_call_and_return_conditional_losses_25002
D__inference_dropout_2_layer_call_and_return_conditional_losses_25014?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dropout_2_layer_call_fn_25019
)__inference_dropout_2_layer_call_fn_25024?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_25040?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_Conv1D_4_layer_call_fn_25049?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_25055?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_flatten_layer_call_fn_25060?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Dense_1_layer_call_and_return_conditional_losses_25071?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_Dense_1_layer_call_fn_25080?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Dense_2_layer_call_and_return_conditional_losses_25091?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_Dense_2_layer_call_fn_25100?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_24623input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
C__inference_Conv1D_1_layer_call_and_return_conditional_losses_24884f4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????	
? ?
(__inference_Conv1D_1_layer_call_fn_24893Y4?1
*?'
%?"
inputs??????????
? "???????????	?
C__inference_Conv1D_2_layer_call_and_return_conditional_losses_24936d#$3?0
)?&
$?!
inputs?????????J	
? ")?&
?
0?????????G
? ?
(__inference_Conv1D_2_layer_call_fn_24945W#$3?0
)?&
$?!
inputs?????????J	
? "??????????G?
C__inference_Conv1D_3_layer_call_and_return_conditional_losses_24988d123?0
)?&
$?!
inputs?????????#
? ")?&
?
0?????????.
? ?
(__inference_Conv1D_3_layer_call_fn_24997W123?0
)?&
$?!
inputs?????????#
? "??????????.?
C__inference_Conv1D_4_layer_call_and_return_conditional_losses_25040d?@3?0
)?&
$?!
inputs?????????.
? ")?&
?
0?????????
? ?
(__inference_Conv1D_4_layer_call_fn_25049W?@3?0
)?&
$?!
inputs?????????.
? "???????????
B__inference_Dense_1_layer_call_and_return_conditional_losses_25071\IJ/?,
%?"
 ?
inputs?????????~
? "%?"
?
0?????????t
? z
'__inference_Dense_1_layer_call_fn_25080OIJ/?,
%?"
 ?
inputs?????????~
? "??????????t?
B__inference_Dense_2_layer_call_and_return_conditional_losses_25091\OP/?,
%?"
 ?
inputs?????????t
? "%?"
?
0?????????!
? z
'__inference_Dense_2_layer_call_fn_25100OOP/?,
%?"
 ?
inputs?????????t
? "??????????!?
I__inference_MaxPooling1D_1_layer_call_and_return_conditional_losses_24016?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
.__inference_MaxPooling1D_1_layer_call_fn_24022wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
I__inference_MaxPooling1D_2_layer_call_and_return_conditional_losses_24031?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
.__inference_MaxPooling1D_2_layer_call_fn_24037wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
I__inference_MaxPooling1D_3_layer_call_and_return_conditional_losses_24046?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
.__inference_MaxPooling1D_3_layer_call_fn_24052wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
 __inference__wrapped_model_24007x#$12?@IJOP5?2
+?(
&?#
input_1??????????
? "1?.
,
Dense_2!?
Dense_2?????????!?
D__inference_dropout_1_layer_call_and_return_conditional_losses_24950d7?4
-?*
$?!
inputs?????????#
p 
? ")?&
?
0?????????#
? ?
D__inference_dropout_1_layer_call_and_return_conditional_losses_24962d7?4
-?*
$?!
inputs?????????#
p
? ")?&
?
0?????????#
? ?
)__inference_dropout_1_layer_call_fn_24967W7?4
-?*
$?!
inputs?????????#
p 
? "??????????#?
)__inference_dropout_1_layer_call_fn_24972W7?4
-?*
$?!
inputs?????????#
p
? "??????????#?
D__inference_dropout_2_layer_call_and_return_conditional_losses_25002d7?4
-?*
$?!
inputs?????????.
p 
? ")?&
?
0?????????.
? ?
D__inference_dropout_2_layer_call_and_return_conditional_losses_25014d7?4
-?*
$?!
inputs?????????.
p
? ")?&
?
0?????????.
? ?
)__inference_dropout_2_layer_call_fn_25019W7?4
-?*
$?!
inputs?????????.
p 
? "??????????.?
)__inference_dropout_2_layer_call_fn_25024W7?4
-?*
$?!
inputs?????????.
p
? "??????????.?
B__inference_dropout_layer_call_and_return_conditional_losses_24898d7?4
-?*
$?!
inputs?????????J	
p 
? ")?&
?
0?????????J	
? ?
B__inference_dropout_layer_call_and_return_conditional_losses_24910d7?4
-?*
$?!
inputs?????????J	
p
? ")?&
?
0?????????J	
? ?
'__inference_dropout_layer_call_fn_24915W7?4
-?*
$?!
inputs?????????J	
p 
? "??????????J	?
'__inference_dropout_layer_call_fn_24920W7?4
-?*
$?!
inputs?????????J	
p
? "??????????J	?
B__inference_flatten_layer_call_and_return_conditional_losses_25055\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????~
? z
'__inference_flatten_layer_call_fn_25060O3?0
)?&
$?!
inputs?????????
? "??????????~?
@__inference_model_layer_call_and_return_conditional_losses_24545t#$12?@IJOP=?:
3?0
&?#
input_1??????????
p 

 
? "%?"
?
0?????????!
? ?
@__inference_model_layer_call_and_return_conditional_losses_24586t#$12?@IJOP=?:
3?0
&?#
input_1??????????
p

 
? "%?"
?
0?????????!
? ?
@__inference_model_layer_call_and_return_conditional_losses_24706s#$12?@IJOP<?9
2?/
%?"
inputs??????????
p 

 
? "%?"
?
0?????????!
? ?
@__inference_model_layer_call_and_return_conditional_losses_24810s#$12?@IJOP<?9
2?/
%?"
inputs??????????
p

 
? "%?"
?
0?????????!
? ?
%__inference_model_layer_call_fn_24241g#$12?@IJOP=?:
3?0
&?#
input_1??????????
p 

 
? "??????????!?
%__inference_model_layer_call_fn_24504g#$12?@IJOP=?:
3?0
&?#
input_1??????????
p

 
? "??????????!?
%__inference_model_layer_call_fn_24839f#$12?@IJOP<?9
2?/
%?"
inputs??????????
p 

 
? "??????????!?
%__inference_model_layer_call_fn_24868f#$12?@IJOP<?9
2?/
%?"
inputs??????????
p

 
? "??????????!?
#__inference_signature_wrapper_24623?#$12?@IJOP@?=
? 
6?3
1
input_1&?#
input_1??????????"1?.
,
Dense_2!?
Dense_2?????????!