<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tff.FunctionType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="parameter"/>
<meta itemprop="property" content="result"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="is_assignable_from"/>
</div>

# tff.FunctionType

## Class `FunctionType`

Inherits From: [`Type`](../tff/Type.md)

An implementation of Type for representing functional types in TFF.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    parameter,
    result
)
```

Constructs a new instance from the given parameter and result types.

#### Args:

* <b>`parameter`</b>: A specification of the parameter type, either an instance of
    Type or something convertible to it by to_type().
* <b>`result`</b>: A specification of the result type, either an instance of
    Type or something convertible to it by to_type().



## Properties

<h3 id="parameter"><code>parameter</code></h3>



<h3 id="result"><code>result</code></h3>





## Methods

<h3 id="is_assignable_from"><code>is_assignable_from</code></h3>

``` python
is_assignable_from(other)
```




