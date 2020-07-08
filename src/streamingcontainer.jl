### StreamingContainer ###

checknames(axnames, ::Type{P}) where {P} = checknames(axnames, dimnames(P))
@noinline function checknames(axnames, parentnames::Tuple{Symbol,Vararg{Symbol}})
    mapreduce(x->in(x, parentnames), &, axnames) || throw(DimensionMismatch("names $axnames are not included among $parentnames"))
    nothing
end

"""
    A = StreamingContainer{T}(f!, parent, streamingaxes::Axis...)

An array-like object possessing one or more axes for which changing "slices" may
be expensive or subject to restrictions. A canonical example would be
an AVI stream, where addressing pixels within the same frame is fast
but jumping between frames might be slow.

Here's a simple example of dividing by the mean of each slice of an image before returning values.

    A = AxisArrays.AxisArray(reshape(1:36, 3, 3, 4))

    function f!(buffer, slice)
        meanslice = mean(slice)
        buffer .= slice./meanslice
    end

    B = StreamingContainer{Float64}(f!, A, axes(A)[3])

    julia> A[:,:,1]
    3×3 AxisArray{Int64,2,Array{Int64,2},Tuple{Axis{:row,Base.OneTo{Int64}},Axis{:col,Base.OneTo{Int64}}}}:
     1  4  7
     2  5  8
     3  6  9

    julia> B[:,:,1]
    3×3 Array{Float64,2}:
     0.2  0.8  1.4
     0.4  1.0  1.6
     0.6  1.2  1.8

The user-provided `f!` function should take arguments:

    f!(buffer, slice)

Where `buffer` will be an empty array that can hold a slice of your series, and `slice` will hold the current input slice.

It's worth noting that `StreamingContainer` is *not* a subtype of
`AbstractArray`, but that much of the array interface (`eltype`,
`ndims`, `axes`, `size`, `getindex`, and `IndexStyle`) is
supported. A StreamingContainer `A` can be built from an AxisArray,
but it may also be constructed from other "parent" objects, even
non-arrays, as long as they support the same functions. In either
case, the parent should also support the standard AxisArray functions
`axes`, `dimnames`, `axisvalues`, and `axisdim`; this support will be
extended to the `StreamingContainer`.

Additionally, a StreamingContainer `A` supports

    getindex!(dest, A, axt::Axis{:time}, ...)

to obtain slices along the streamed axes (here it is assumed that
`:time` is a streamed axis of `A`). You can implement this directly
(dispatching on the parameters of `A`), or (if the parent is an
`AbstractArray`) rely on the fallback

    A.getindex!(dest, view(parent, axs...))

where `A.getindex! = f!` as passed as an argument at construction. `dest` should
have dimensionality `ndims(parent)-length(streamingaxes)`.

Optionally, define [`StreamIndexStyle(typeof(parent),typeof(f!))`](@ref).
"""
struct StreamingContainer{L,T,N,P,GetIndex}
    getindex!::GetIndex
    parent::P
end

function StreamingContainer{L,T}(f!::Function, parent) where {L,T}
    N = ndims(parent)
    checknames(L, typeof(parent))
    return StreamingContainer{L,T,N,typeof(parent),typeof(f!)}(f!, parent)
end

Base.parent(s::StreamingContainer) = getfield(s, :parent)
Base.axes(s::StreamingContainer) = axes(parent(s))
Base.size(s::StreamingContainer)    = size(parent(s))
Base.axes(s::StreamingContainer, d) = axes(parent(s), d)
Base.size(s::StreamingContainer, d)    = size(parent(s), d)


Base.eltype(::Type{<:StreamingContainer{L,T,N}}) where {L,T,N} = T
Base.ndims(::Type{<:StreamingContainer{L,T,N}}) where {L,T,N} = N
Base.length(S::StreamingContainer) = prod(size(S))

AxisIndices.dimnames(::T) where {T<:StreamingContainer} = dimnames(T)
function AxisIndices.dimnames(::Type{<:StreamingContainer{L,T,N,P}}) where {L,T,N,P}
    return dimnames(P)
end

AxisIndices.has_dimnames(::Type{<:StreamingContainer}) = true

streaming_dimnames(::T) where {T} = streaming_dimnames(T)
streaming_dimnames(::Type{<:StreamingContainer{L}}) where {L} = L

is_streamdim(name::Symbol, s::StreamingContainer) = name in streaming_dimnames(s)

@inline function getindex!(s::StreamingContainer{T,N}; named_inds...) where {T,N}
    inds = AxisIndices.Interface.NamedDims.order_named_inds(Val(dimnames(s)); named_inds...)
    return getindex!(s, inds...)
end

#=
@inline function getindex!(dest, s::StreamingContainer, axs...)
    if any(ax->!is_streamdim(ax, s), axs)
        throw(ArgumentError("$axs do not coincide with the streaming axes $(streaming_dimnames(S))"))
    end
    return _getindex!(dest, s.getindex!, parent(s), axs...)
end
=#

@inline function getindex!(dest, s::StreamingContainer, I::Vararg{Any,N}) where {N}
    axs = getslicedindices(s, I)
    ntuple(Val(N)) do i
        if !is_streamdim(dimnames(s, i), dimnames(s)) && !isa(getfield(I, i), Colon)
            throw(ArgumentError("must use `:` for any non-streaming axes"))
        end
    end
    return _getindex!(dest, s.getindex!, parent(s), axs...)
end

# _getindex! just makes it easy to specialize for particular parent or function types
@inline _getindex!(dest, f!, P, axs...) = f!(dest, view(P, axs...))

sliceindices(S::StreamingContainer) = filter_notstreamed(axes(S), S)
sliceaxes(S::StreamingContainer) = filter_notstreamed(axes(S), S)
function getslicedindices(S::StreamingContainer, I)
    return filter_streamed(map((ax, i) -> ax(i), axes(S), I), S)
end

filter_notstreamed(inds, S) = _filter_notstreamed(inds, named_axes(S), S)
filter_streamed(inds::Tuple{Axis,Vararg{Axis}}, S)    = _filter_streamed(inds, inds, S)
filter_notstreamed(inds::Tuple{Axis,Vararg{Axis}}, S) = _filter_notstreamed(inds, inds, S)

@generated function _filter_streamed(a, axs::NTuple{N,Axis}, S::StreamingContainer) where N
    inds = findall(x-> x in streaming_dimnames(S), dimnames(axs.parameters...))
    Expr(:tuple, Expr[:(a[$i]) for i in inds]...)
end
@generated function _filter_notstreamed(a, axs::NTuple{N,Axis}, S::StreamingContainer) where N
    inds = findall(x->x in streaming_dimnames(S), dimnames(axs.parameters...))
    inds = setdiff(1:N, inds)
    Expr(:tuple, Expr[:(a[$i]) for i in inds]...)
end

# ::Vararg{Union{Colon,Base.ViewIndex},N}
@inline function Base.getindex(s::StreamingContainer{T,N}, inds...) where {T,N}
    tmp = similar(Array{T}, sliceindices(s))
    getindex!(tmp, s, getslicedindices(S, inds)...)
    tmp[filter_notstreamed(inds, s)...]
end

@inline function Base.getindex(s::StreamingContainer{T,N}; named_inds...) where {T,N}
    inds = AxisIndices.Interface.NamedDims.order_named_inds(Val(dimnames(s)); named_inds...)
    return getindex(s, inds...)
end

@inline function Base.getindex(S::StreamingContainer, ind1::Axis, inds_rest::Axis...)
    axs = sliceaxes(S)
    tmp = AxisArray(Array{eltype(S)}(undef, map(length, axs)), axs)
    inds = (ind1, inds_rest...)
    getindex!(tmp, S, _filter_streamed(inds, inds, S)...)
    getindex_rest(tmp, _filter_notstreamed(inds, inds, S))
end
getindex_rest(tmp, ::Tuple{}) = tmp
getindex_rest(tmp, inds) = tmp[inds...]

"""
    style = StreamIndexStyle(A)

A trait that indicates the degree of support for indexing the streaming axes
of `A`. Choices are [`IndexAny()`](@ref) and
[`IndexIncremental()`](@ref) (for arrays that only permit advancing
the time axis, e.g., a video stream from a webcam). The default value
is `IndexAny()`.

This should be specialized for the type rather than the instance. For
a StreamingContainer `S`, you can define this trait via

```julia
StreamIndexStyle(::Type{P}, ::typeof(f!)) = IndexIncremental()
```

where `P = typeof(parent(S))`.
"""
abstract type StreamIndexStyle end

"""
    IndexAny()

Indicates that an axis supports full random-access indexing.
"""
struct IndexAny <: StreamIndexStyle end

"""
    IndexIncremental()

Indicates that an axis supports only incremental indexing, i.e., from `i` to `i+1`.
This is commonly used for the temporal axis with media streams.
"""
struct IndexIncremental <: StreamIndexStyle end

StreamIndexStyle(::Type{A}) where {A<:AbstractArray} = IndexAny()
StreamIndexStyle(A::AbstractArray) = StreamIndexStyle(typeof(A))

function StreamIndexStyle(
    ::Type{StreamingContainer{T,N,axnames,P,GetIndex}}
) where {T,N,axnames,P,GetIndex}

    return StreamIndexStyle(P, GetIndex)
end

StreamIndexStyle(::Type{P},::Type{GetIndex}) where {P,GetIndex} = IndexAny()

StreamIndexStyle(S::StreamingContainer) = StreamIndexStyle(typeof(S))

### Low level utilities ###

#=
filter_space_axes(axes::NTuple{N,Axis}, items::NTuple{N,Any}) where {N} =
    _filter_space_axes(axes, items)
@inline @traitfn _filter_space_axes(axes::Tuple{Ax,Vararg{Any}}, items) where {Ax<:Axis;  TimeAxis{Ax}} =
    _filter_space_axes(tail(axes), tail(items))
@inline @traitfn _filter_space_axes(axes::Tuple{Ax,Vararg{Any}}, items) where {Ax<:Axis; !TimeAxis{Ax}} =
    (items[1], _filter_space_axes(tail(axes), tail(items))...)
_filter_space_axes(::Tuple{}, ::Tuple{}) = ()
@inline _filter_space_axes(axes::Tuple{Ax,Vararg{Any}}, items) where {Ax<:Axis{:color}} =
    _filter_space_axes(tail(axes), tail(items))

filter_time_axis(axes::NTuple{N,Axis}, items::NTuple{N}) where {N} =
    _filter_time_axis(axes, items)
@inline @traitfn _filter_time_axis(axes::Tuple{Ax,Vararg{Any}}, items) where {Ax<:Axis; !TimeAxis{Ax}} =
    _filter_time_axis(tail(axes), tail(items))
@inline @traitfn _filter_time_axis(axes::Tuple{Ax,Vararg{Any}}, items) where {Ax<:Axis;  TimeAxis{Ax}} =
    (items[1], _filter_time_axis(tail(axes), tail(items))...)
_filter_time_axis(::Tuple{}, ::Tuple{}) = ()

# summary: print color types & fixed-point types compactly
function AxisArrays._summary(io, A::AxisArray{T,N}) where {T<:Union{Fractional,Colorant},N}
    print(io, "$N-dimensional AxisArray{")
    if T<:Colorant
        ColorTypes.colorant_string_with_eltype(io, T)
    else
        ColorTypes.showcoloranttype(io, T)
    end
    println(io, ",$N,...} with axes:")
end

=#
