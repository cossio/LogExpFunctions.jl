@testset "ChainRules numerical stability" begin
    @testset "softmax collapse" begin
        # The reference must retain 1 - p_max even when p_max rounds to one
        # in the input precision. Finite differences cannot resolve this case.
        for (T, a) in ((Float16, 6), (Float32, 9), (Float64, 20))
            for k in 1:3
                x = fill(-T(a), 3)
                x[k] = T(a)
                basis = zeros(T, 3)
                basis[k] = one(T)
                for v in (basis, -basis, T[2, -1, 3])
                    expected = setprecision(BigFloat, 256) do
                        p = exp.(BigFloat.(x))
                        p ./= sum(p)
                        T.(p .* (BigFloat.(v) .- sum(p .* BigFloat.(v))))
                    end
                    cases = (
                        (x, v, expected, :),
                        (x, v, expected, 1),
                        (repeat(x, 1, 2), repeat(v, 1, 2), repeat(expected, 1, 2), 1),
                        (repeat(permutedims(x), 2, 1), repeat(permutedims(v), 2, 1),
                         repeat(permutedims(expected), 2, 1), 2),
                        (reshape(x, 1, 3), reshape(v, 1, 3), reshape(expected, 1, 3), 1:2),
                        (reshape(x, 3, 1, 1), reshape(v, 3, 1, 1), reshape(expected, 3, 1, 1), (1, 3)),
                        (OffsetArray(x, -1:1), OffsetArray(v, -1:1), OffsetArray(expected, -1:1), :),
                        (OffsetArray(x, -1:1), OffsetArray(v, -1:1), OffsetArray(expected, -1:1), 1),
                    )
                    for (X, V, E, dims) in cases
                        y, dy = frule((NoTangent(), V), softmax, X; dims=dims)
                        y_reverse, back = rrule(softmax, X; dims=dims)
                        thunk = back(V)[2]
                        dx = unthunk(thunk)
                        @test y == y_reverse == softmax(X; dims=dims)
                        @test axes(dy) == axes(dx) == axes(X)
                        @test eltype(dy) == eltype(dx) == T
                        @test dy ≈ E rtol=8eps(T) atol=zero(T)
                        @test dx ≈ E rtol=8eps(T) atol=zero(T)
                        @test unthunk(back(@thunk(V))[2]) ≈ E rtol=8eps(T) atol=zero(T)
                        # Check relative conservation; an absolute tolerance
                        # would accept the original all-but-one gradient.
                        @test all(abs.(sum(dx; dims=dims)) .<=
                                  8eps(T) .* sum(abs, dx; dims=dims))
                        destination = copy(E)
                        @test thunk.add!(destination) === destination
                        @test destination ≈ 2 .* E rtol=8eps(T) atol=zero(T)
                        # The returned primal is mutable; pullbacks must keep
                        # their own probability values and remain reusable.
                        fill!(y_reverse, zero(T))
                        @test unthunk(back(V)[2]) ≈ E rtol=8eps(T) atol=zero(T)
                    end
                end
            end
        end
    end

    @testset "softmax near exponential underflow" begin
        for (T, a) in ((Float16, 8), (Float32, 40), (Float64, 350))
            x = T[a, -a, -a]
            v = T[1, 0, 0]
            # This expression avoids subtracting from one even in BigFloat.
            r = exp(-2big(a))
            derivative = r / (1 + 2r)^2
            expected = T.([2derivative, -derivative, -derivative])
            @test expected[1] > 0
            @test frule((NoTangent(), v), softmax, x)[2] ≈ expected rtol=8eps(T) atol=zero(T)
            @test unthunk(rrule(softmax, x)[2](v)[2]) ≈ expected rtol=8eps(T) atol=zero(T)
        end
    end

    @testset "softmax ties and constant tangents" begin
        for x in ([0.0, 0.0, -1.0], [2.0], [9.0, -9.0, -9.0]), dims in (:, 1, 2, ())
            v = ones(size(x))
            @test all(iszero, frule((NoTangent(), v), softmax, x; dims=dims)[2])
            @test all(iszero, unthunk(rrule(softmax, x; dims=dims)[2](v)[2]))
            @test frule((NoTangent(), ZeroTangent()), softmax, x; dims=dims)[2] isa ZeroTangent
            @test rrule(softmax, x; dims=dims)[2](ZeroTangent())[2] isa ZeroTangent
        end
        x = [0.0, 0.0, -1.0]
        test_frule(softmax, x)
        test_rrule(softmax, x)
    end

    @testset "logistic saturation" begin
        for (T, a) in ((Float16, 12), (Float32, 18), (Float64, 40),
                       (Float16, 17), (Float32, 103.5), (Float64, 745)), sign in (-1, 1)
            x = T(sign * a)
            expected = setprecision(BigFloat, 256) do
                e = exp(-BigFloat(x))
                T(e / (1 + e)^2)
            end
            @test expected > 0
            y, dy = frule((NoTangent(), one(T)), logistic, x)
            y_reverse, back = rrule(logistic, x)
            @test y == y_reverse == logistic(x)
            @test dy ≈ expected rtol=4eps(T) atol=zero(T)
            @test back(one(T))[2] ≈ expected rtol=4eps(T) atol=zero(T)
        end
        for x in (-Inf, Inf)
            @test iszero(frule((NoTangent(), 1.0), logistic, x)[2])
            @test iszero(rrule(logistic, x)[2](1.0)[2])
        end
        @test isnan(rrule(logistic, NaN)[2](1.0)[2])
        @test rrule(logistic, 0)[2](1.0)[2] == 0.25
        @test iszero(rrule(logistic, typemin(Int))[2](1.0)[2])
    end

    @testset "logsumexp common offsets" begin
        for (T, a) in ((Float16, 2048), (Float32, 1e8), (Float64, 1e16))
            for (x, dims) in ((fill(T(a), 2), :), (fill(T(a), 2, 3), 1),
                              (fill(T(a), 3, 2), 2), (fill(T(a), 1, 2), 1:2),
                              (OffsetArray(fill(T(a), 2), -1:0), 1))
                # Two equal logits have weights 1/2 regardless of their offset.
                tangent = fill!(similar(x), one(T))
                y, dy = frule((NoTangent(), tangent), logsumexp, x; dims=dims)
                y_reverse, back = rrule(logsumexp, x; dims=dims)
                seed = dims === (:) ? one(T) : one.(y_reverse)
                expected = fill!(similar(x), T(0.5))
                thunk = back(seed)[2]
                @test y_reverse == logsumexp(x; dims=dims)
                @test all(isone, dy)
                @test unthunk(thunk) == expected
                destination = copy(expected)
                @test thunk.add!(destination) === destination
                @test destination == 2 .* expected
            end
        end
        # Empty inputs were supported by the existing logsumexp rules.
        x = Float64[]
        y, dy = frule((NoTangent(), Float64[]), logsumexp, x)
        @test y == -Inf
        @test iszero(dy)
        @test isempty(unthunk(rrule(logsumexp, x)[2](1.0)[2]))
        # Normalize floating-point inputs so integer subtraction cannot wrap.
        for (x, expected) in ((UInt[1, 2], [1 / (1 + exp(1)), 1 / (1 + exp(-1))]),
                              ([typemin(Int), typemax(Int)], [0.0, 1.0]))
            @test unthunk(rrule(logsumexp, x)[2](1.0)[2]) ≈ expected
            @test frule((NoTangent(), [2.0, -1.0]), logsumexp, x)[2] ≈ 2expected[1] - expected[2]
        end
    end
end
