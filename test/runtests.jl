using ExplicitQPs
using Base.Test
using JuMP
using Gurobi


@testset "trivial model" begin
    m = Model(solver=GurobiSolver(OutputFlag=0))
    @variable m x >= 0.5
    @variable m y
    @constraint m y >= x
    JuMP.fix(x, 0.5)
    @objective m Min x^2 + y^2 + 0.1 * x * y
    solve(m)
    @test getvalue(y) ≈ 0.5
    @test getvalue(x) ≈ 0.5
    ex = ExplicitQPs.explicit_solution(m, [x])
    s = ExplicitQPs.solution(ex, y)
    @test getvalue(s) ≈ 0.5
    @test ExplicitQPs.parameter(ex, x) == getvalue(x)
    @test ExplicitQPs.gradient(ex, y) ≈ [1.0]
    @test ExplicitQPs.jacobian(ex, [y]) ≈ [1.0]
end

@testset "models with affine cost" begin
    @testset "model 1" begin
        m = Model(solver=GurobiSolver(OutputFlag=0))
        @variable m x
        @variable m y
        @constraint m y <= x
        JuMP.fix(x, 0.5)
        @objective m Min (y - 0.6)^2
        solve(m)
        @test getvalue(y) ≈ 0.5
        @test getvalue(x) ≈ 0.5
        ex = ExplicitQPs.explicit_solution(m, [x])
        s = ExplicitQPs.solution(ex, y)
        @test getvalue(s) ≈ 0.5
        @test ExplicitQPs.parameter(ex, x) == getvalue(x)
        @test ExplicitQPs.gradient(ex, y) ≈ [1.0]
        @test ExplicitQPs.jacobian(ex, [y]) ≈ [1.0]
    end

    @testset "model 2" begin
        m = Model(solver=GurobiSolver(OutputFlag=0))
        @variable m x
        @variable m y
        @constraint m y <= 10
        JuMP.fix(x, 0.5)
        @objective m Min (y - 0.6)^2
        solve(m)
        @test getvalue(y) ≈ 0.6
        @test getvalue(x) ≈ 0.5
        ex = ExplicitQPs.explicit_solution(m, [x])
        s = ExplicitQPs.solution(ex, y)
        @test getvalue(s) ≈ 0.6
        @test ExplicitQPs.parameter(ex, x) == getvalue(x)
        @test ExplicitQPs.gradient(ex, y) ≈ [0.0]
        @test ExplicitQPs.jacobian(ex, [y]) ≈ [0.0]
    end
end

@testset "simple mpc" begin
    env = Gurobi.Env()
    function run_mpc(x0)
        m = Model(solver=GurobiSolver(env, OutputFlag=0))
        @variable(m, x[1:2])
        JuMP.fix.(x, x0)
        U = []
        X = [x]
        N = 3
        Δt = 0.1
        for i in 1:N
            u = @variable(m, lowerbound=-1, upperbound=1)
            xi = @variable(m, [1:2])
            @constraint(m, xi[2] == X[end][2] + Δt * u - 0.1)
            @constraint(m, xi[1] == X[end][1] + Δt * xi[2])
            push!(U, u)
            push!(X, xi)
        end

        @objective m Min sum([x[1]^2 + 0.01 * x[2]^2 for x in X]) + 0.01 * sum([(u - 0.5)^2 for u in U])
        solve(m)
        m, X, U
    end

    srand(1)
    correct = 0
    incorrect = 0
    for i in 1:100
        x0 = 0.1 .* randn(2)
        m, X, U = run_mpc(x0);
        ex = ExplicitQPs.explicit_solution(m, X[1]);
        s = ExplicitQPs.solution(ex, U[1])
        J = ExplicitQPs.jacobian(ex, [U[1]])

        if isapprox(getvalue(s), getvalue(U[1]), atol=1e-3, rtol=1e-3)
            correct += 1
        else
            incorrect += 1
        end

        for j in 1:2
            Δx = zeros(2)
            eps = 1e-9
            Δx[j] = eps
            m2, X2, U2 = run_mpc(x0 .+ Δx)
            if isapprox((getvalue(U2[1]) - getvalue(U[1])) / eps, J[1, j], rtol=1e-4)
                correct += 1
            else
                incorrect += 1
            end
        end
    end
    @test (correct / (correct + incorrect)) >= 0.85
end
