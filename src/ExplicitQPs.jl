__precompile__()

module ExplicitQPs

using JuMP
using JuMP: GenericAffExpr, AffExpr

# See Bemporad, "A Survey on Explicit Model Predictive Control",
# section 2.2 (page 353)

getcol(x::Variable) = x.col

struct ExplicitSolution{T, M <: AbstractMatrix{T}}
    model::Model
    solution::Vector{GenericAffExpr{T, Variable}}
    constant::Vector{T}
    jacobian::M
    params::Vector{T}
    variable_map::Vector{Tuple{Bool, Int}}
end

function solution(ex::ExplicitSolution, v::Variable)
    @assert v.m === ex.model
    isparam, idx = ex.variable_map[getcol(v)]
    @assert !isparam "$v is a parameter, so you can access its value with getparameter()"
    ex.solution[idx]
end

function parameter(ex::ExplicitSolution, v::Variable)
    @assert v.m === ex.model
    isparam, idx = ex.variable_map[getcol(v)]
    @assert isparam "$v is not a parameter, so you can access its value with getsolution()"
    ex.params[idx]
end

function gradient(ex::ExplicitSolution, v::Variable)
    @assert v.m === ex.model
    isparam, idx = ex.variable_map[getcol(v)]
    @assert !isparam "cannot extract a jacobian for a parameter"
    ex.jacobian[idx, :]
end

function jacobian(ex::ExplicitSolution, vars::AbstractVector{Variable})
    hcat(gradient.(ex, vars)...)'
end

function variable_map(m::Model, params::AbstractArray{Variable})
    var_index = 1
    param_index = 1
    nvars = length(m.colCat)
    var_map = Tuple{Bool, Int}[]
    for i in 1:nvars
        var = Variable(m, i)
        if var in params
            push!(var_map, (true, param_index))
            param_index += 1
        else
            push!(var_map, (false, var_index))
            var_index += 1
        end
    end
    @assert param_index == (length(params) + 1)
    @assert param_index + var_index == (nvars + 2)
    var_map
end

function active_inequalities(m::Model, params::AbstractArray{Variable}, eps=1e-6)
    nvars = length(m.colCat)
    A_active = SparseVector{Float64, Int64}[]
    b_active = Float64[]

    for i in 1:nvars
        var = Variable(m, i)
        if var in params
            continue
        end
        λ = getdual(var)
        if abs(λ) > eps
            if λ > 0
                ai = sparsevec([i], [-1.0], nvars)
                bi = -m.colLower[i]
            else
                ai = sparsevec([i], [1.0], nvars)
                bi = m.colUpper[i]
            end
            push!(A_active, ai)
            push!(b_active, bi)
        end
    end

    nconstr = length(m.linconstr)
    for (constraint, λ) in zip(m.linconstr, m.linconstrDuals)
        if abs(λ) > eps
            if λ > 0
                ai = -sparsevec([var.col for var in constraint.terms.vars], constraint.terms.coeffs, nvars)
                bi = -constraint.lb
            else
                ai = sparsevec([var.col for var in constraint.terms.vars], constraint.terms.coeffs, nvars)
                bi = constraint.ub
            end
            push!(A_active, ai)
            push!(b_active, bi)
        end
    end

    A::SparseMatrixCSC{Float64,Int} = if isempty(A_active)
        sparse(zeros(0, nvars))
    else
        hcat(A_active...)'
    end
    b::Vector{Float64} = vcat(b_active...)
    A, b
end

# Handling affine cost terms
# Let's say we have a problem of the form:
# min. 1/2 u' H u + x' F u + q' u
#  u
# s.t. G u <= W + S x
#
# We can perform a change of variables:
# let u = y + v
# where v is a constant. Then the problem becomes:
# min 1/2 (y + v)' H (y + v) + x' F (y + v) + q' (y + v)
#  y
# s.t. G (y + v) <= W + S x
#
# We can rewrite the cost as:
# min 1/2 y' H y + x' F y + q' y + v' H y + 1/2 v' H v + x' F v + q' v
#  y
# s.t. G y <= W - G v + S x
#
# and we can drop constants from the cost to get:
# min 1/2 y' H y + x' F y + q' y + v' H y
#
# If we choose v such that
# q' y + v' H y = 0
#
# then we end up with a problem in the standard explicit QP form:
# min 1/2 y' H y + x' F y
#  y
# s.t. G y <= (W - G v) + S x
#
# To do this, we need:
# q' = -v' H
# or
# H' v = -q
#
# So the procedure is:
# Solve the explicit QP with H, F, G and S unchanged, but with
# W <- W - G v
# where v = -H' \ q
# then substitute u = y + v when returning the answer

function explicit_solution(m::Model, params::AbstractArray{Variable}, eps=1e-3)
    nvars = length(m.colCat)
    Ã, W̃ = active_inequalities(m, params)

    param_cols = Set([v.col for v in params])
    isparam = collect(1:nvars) .∈ param_cols

    G̃ = Ã[:, .!isparam]
    S̃ = .-Ã[:, isparam]

    Q = sparse(getcol.(m.obj.qvars1), getcol.(m.obj.qvars2), m.obj.qcoeffs, nvars, nvars)
    Q = (Q .+ Q')

    # todo: cholesky instead?
    H = lufact(Q[.!isparam, .!isparam])

    # Handle affine cost terms by performing change of variables
    # to solve for (y + v) instead of u
    if !isempty(m.obj.aff)
        q = zeros(nvars)
        for i in eachindex(m.obj.aff.vars)
            var, coeff = m.obj.aff.vars[i], m.obj.aff.coeffs[i]
            q[var.col] += coeff
        end
        q̃ = q[.!isparam]
        ṽ = -(H \ q̃)
    else
        ṽ = zeros(size(G̃, 2))
    end
    W̃ .-= G̃ * ṽ

    F = Q[.!isparam, isparam]

    # v_test = rand(nvars)
    # x_test = v_test[isparam]
    # z_test = v_test[.!isparam]
    # Y = Q[isparam, isparam]
    # @assert 0.5 * v_test' * Q * v_test ≈ (0.5 * z_test' * H * z_test + x_test' * F' * z_test + 0.5 * x_test' * Y * x_test)

    x = getvalue(params)
    HiG = H \ full(G̃')
    HiF = H \ full(F)
    λ_active = -(G̃ * HiG) * (W̃ + (S̃ + G̃ * HiF) * x)

    GHiG = lufact(G̃ * HiG)

    jacobian = HiG * (GHiG \ (S̃ + G̃ * HiF)) - HiF
    constant = HiG * (GHiG \ full(W̃))

    # Undo change of variables from u to y
    constant .+= ṽ

    solution = [AffExpr(params, jacobian[i, :], constant[i]) for i in 1:length(constant)]
    @assert isapprox(JuMP.getvalue.(solution), constant + jacobian * x)

    ExplicitSolution(m, solution, constant, jacobian, x, variable_map(m, params))
end

end