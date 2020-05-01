"""
    HasProperties(img) -> HasProperties{::Bool}

Returns the trait `HasProperties`, indicating whether `x` has `properties`
method.
"""
struct HasProperties{T} end

HasProperties(img::T) where T = HasProperties(T)

HasProperties(::Type{T}) where T = HasProperties{false}()

"""
    HasDimNames(img) -> HasDimNames{::Bool}

Returns the trait `HasDimNames`, indicating whether `x` has named dimensions.
Types returning `HasDimNames{true}()` should also have a `names` method that
returns a tuple of symbols for each dimension.
"""
struct HasDimNames{T} end

HasDimNames(img::T) where T = HasDimNames(T)

HasDimNames(::Type{T}) where T = HasDimNames{false}()

"""
    namedaxes(img) -> NamedTuple{names}(axes)

Returns a `NamedTuple` where the names are the dimension names and each indice
is the corresponding dimensions's axis. If `HasDimNames` is not defined for `x`
default names are returned. `x` should have an `axes` method.

```jldoctest; setup = :(using ImageCore)
julia> img = reshape(1:24, 2,3,4);

julia> namedaxes(img)
(dim_1 = Base.OneTo(2), dim_2 = Base.OneTo(3), dim_3 = Base.OneTo(4))
```
"""
namedaxes(img::T) where T = namedaxes(HasDimNames(T), img)

namedaxes(::HasDimNames{true}, x::T) where T = NamedTuple{names(x)}(axes(x))

function namedaxes(::HasDimNames{false}, img::AbstractArray{T,N}) where {T,N}
    NamedTuple{default_names(Val(N))}(axes(img))
end

@generated function default_names(img::Val{N}) where {N}
    :($(ntuple(i -> Symbol(:dim_, i), N)))
end

"""
    nimages(img)

Return the number of time-points in the image array. Use `NamedDims` if you want
to use an explicit time dimension.
"""
nimages(img::AbstractArray) = ntime(img)

#### Utilities for writing "simple algorithms" safely ####
# If you don't feel like supporting multiple representations, call these
widthheight(img::AbstractArray) = length(axes(img,2)), length(axes(img,1))

width(img::AbstractArray) = widthheight(img)[1]
height(img::AbstractArray) = widthheight(img)[2]

# Utilities

@inline traititer(f, A, rest...) = (f(A), traititer(f, rest...)...)
@inline traititer(f, A::ZeroArray, rest...) = traititer(f, rest...)
traititer(f) = ()

function checksame(t::Tuple)
    val1 = t[1]
    @assert all(p -> p == val1, t)
    return val1
end

### Start new traits
###
###

Base.@pure is_color(x::Symbol) = x === :color

AxisIndices.@defdim color is_color

###
### Spatial traits
###
# yes, I'm abusing @pure
Base.@pure function is_spatial(x::Symbol)
    return !is_time(x) && !is_color(x) && !is_observation(x)
end

"""
    spatial_order(x) -> Tuple{Vararg{Symbol}}

Returns the `dimnames` of `x` that correspond to spatial dimensions.
"""
spatial_order(x::X) where {X} = _spatial_order(Val(dimnames(X)))
@generated function _spatial_order(::Val{L}) where {L}
    keep_names = []
    for n in L
        if is_spatial(n)
            push!(keep_names, n)
        end
    end
    out = (keep_names...,)
    quote
        return $out
    end
end

"""
    spatialdims(x) -> Tuple{Vararg{Int}}

Return a tuple listing the spatial dimensions of `img`.
Note that a better strategy may be to use ImagesAxes and take slices along the time axis.
"""
@inline spatialdims(x) = dim(dimnames(x), spatial_order(x))

"""
    spatial_axes(x) -> Tuple

Returns a tuple of each axis corresponding to a spatial dimensions.
"""
@inline spatial_axes(x) = _spatial_axes(named_axes(x), spatial_order(x))
function _spatial_axes(na::NamedTuple, spo::Tuple{Vararg{Symbol}})
    return map(spo_i -> getfield(na, spo_i), spo)
end

"""
    spatial_size(img) -> Tuple{Vararg{Int}}

Return a tuple listing the sizes of the spatial dimensions of the
image. Defaults to the same as `size`, but using AxisIndices you can
mark some axes as being non-spatial.
"""
@inline spatial_size(x) = map(length, spatial_axes(x))

"""
    spatial_indices(x)

Return a tuple with the indices of the spatial dimensions of the
image. Defaults to the same as `indices`, but using `NamedDimsArray` you can
mark some axes as being non-spatial.
"""
@inline spatial_indices(x) = map(values, spatial_axes(x))

# TODO document this
"""
    spatial_keys(x)
"""
@inline spatial_keys(x) = map(keys, spatial_axes(x))

# FIXME account for keys with no steps
"""
    pixel_spacing(x)

Return a tuple representing the separation between adjacent pixels along each axis
of the image. Derived from the step size of each element of `spatial_keys`.
"""
@inline function pixel_spacing(x)
    map(spatial_keys(x)) do ks_i
        if StaticRanges.has_step(ks_i)
            return step(ks_i)
        else
            return 0
        end
    end
end

"""
    spatial_offset(x)

The offset of each dimension (i.e., where each spatial axis starts).
"""
spatial_offset(x) = map(first, spatial_keys(x))

"""
    spatial_directions(x) -> (axis1, axis2, ...)

Return a tuple-of-tuples, each `axis[i]` representing the displacement
vector between adjacent pixels along spatial axis `i` of the image
array, relative to some external coordinate system ("physical
coordinates").

By default this is computed from `pixel_spacing`, but you can set this
manually using ImagesMeta.
"""
function spatial_directions(x::AbstractArray{T,N}) where {T,N}
    ntuple(Val(N)) do i
        ntuple(Val(N)) do d
            if d === i
                if is_spatial(dimnames(x, i))
                    ks = axes_keys(x, i)
                    if StaticRanges.has_step(ks)
                        return step(ks)
                    else
                        return 1  # TODO If keys are not range does it make sense to return this?
                    end
                else
                    return 0
                end
            else
                return 0
            end
        end
    end
end

"""
    sdims(x)

Return the number of spatial dimensions in the image. Defaults to the same as
`ndims`, but with `NamedDimsArray` you can specify that some dimensions
correspond to other quantities (e.g., time) and thus not included by `sdims`.
"""
@inline function sdims(x)
    cnt = 0
    for name in dimnames(x)
        if is_spatial(name)
            cnt += 1
        end
    end
    return cnt
end

