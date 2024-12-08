using QuantumCollocation
using NamedTrajectories
using Optim
using TrajectoryIndexingUtils
using LinearAlgebra
using CairoMakie
using JLD2
using Revise
using ForwardDiff

function euler(dda,n_controls::Int64,n_steps::Int64,Δt::Float64)
    da_init=-sum(hcat([0 for i in 1:n_controls],cumsum(dda[:,1:end-1]*Δt,dims=2))[:,1:end-1],dims=2)/(n_steps-1)
    da=hcat([0 for i in 1:n_controls],cumsum(dda[:,1:end-1]*Δt,dims=2)) + reduce(hcat,[da_init for i in 1:n_steps])
    a_=hcat([0 for i in 1:n_controls],cumsum(da[:,1:end-1]*Δt,dims=2))
    return a_
end


function accel_SVD_rollout(dda::Matrix{Float64},system::QuantumSystem, n_steps::Int64,Δt::Float64; weighted_modes::Bool=false)
    dda⃗ = reshape(dda, length(system.H_drives) * n_steps)

    function U⃗(dda⃗)
        dda = reshape(dda⃗, length(system.H_drives), n_steps)
        a = euler(dda,length(system.H_drives),n_steps,Δt)
        return unitary_rollout(a,ones(n_steps)*Δt, system)[:,end]
    end

    _,σ,V = svd(ForwardDiff.jacobian(dda⃗ -> U⃗(dda⃗), dda⃗))
    
    accel_modes = [reshape(V[:,i], length(system.H_drives) , n_steps) for i in range(1,size(V)[end])]
    control_modes = [euler(dda,length(system.H_drives),n_steps,Δt) for dda in accel_modes]

    if(!weighted_modes)
        return control_modes
    end

    σ = σ/(maximum(σ))
    σ = [x<=1e-5 ? 1e-5 : x for x ∈ σ]    
    
    return [c/(maximum(abs.(c))) * σ[i] for (i,c) ∈ enumerate(control_modes)]
end


function plot_data(data::Vector{Matrix{Float64}}; figsize::Tuple{Int64, Int64}=(1200,200))
    f = Figure(size =figsize)
    for idx ∈ 1:length(data)
        ax = Axis(f[1, idx],xlabel="Timestep",ylabel="Amplitude")
        for k ∈ 1:size(data[idx])[1]
            lines!(ax, data[idx][k,:])
        end
    end
    return f
end

function perturb_rollout(base_a::Matrix{Float64},modes::Vector{Matrix{Float64}},perterb_coeffs::Vector{<:Real },
    system::QuantumSystem, n_steps::Int64,Δt::Float64,G::Matrix{ComplexF64})
    a_out = sum([modes[i]*perterb_coeffs[i] for i ∈ 1:length(perterb_coeffs)]) + base_a
    return unitary_rollout(a_out,ones(n_steps)*Δt, system)[:,end]
end

function loss_function(base_a::Matrix{Float64},modes::Vector{Matrix{Float64}},system::QuantumSystem, n_steps::Int64,Δt::Float64,G::Matrix{ComplexF64})
    function loss(a::Vector{<:Real})
        Ũ⃗=  perturb_rollout(base_a,modes,a,system, n_steps,Δt,G)
        return 1-iso_vec_unitary_fidelity(Ũ⃗,operator_to_iso_vec(G))
    end
    return loss
end

function neadler_mead_opt(trunc_level::Int64,base_a::Matrix{Float64},modes::Vector{Matrix{Float64}},system::QuantumSystem, n_steps::Int64,Δt::Float64,G::Matrix{ComplexF64}
    ;iterations::Int64=1000)
    l = loss_function(base_a,modes,system, n_steps,Δt,G)
    x0 = zeros(trunc_level)
    res = optimize(l, x0, NelderMead(),Optim.Options(store_trace = true,iterations=iterations))
    return reduce(vcat,[[l(x0)],[s.value for s ∈ res.trace],[Optim.minimum(res)]]),Optim.minimizer(res)
end


function perturb_system(system::QuantumSystem,ϵ::Vector{Float64})
    H_drift = system.H_drift
    H_drives = [H*(1+ϵ[i]) for (i,H) ∈ enumerate(system.H_drives)]
    return QuantumSystem(H_drift,H_drives)
end

relu(x)=[i<=1e-5 ? 1e-5 : i for i ∈ x]