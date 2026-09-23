using GLMakie
using Statistics
using Printf

function process_observable(file_path::String, R::Int)
    raw_data = collect(reinterpret(Float64, read(file_path)))
    num_records = div(length(raw_data), R)

    mat = reshape(raw_data, (R, num_records))

    mean_vals = vec(mean(mat, dims=1))
    std_vals = vec(std(mat, dims=1))
    sem_vals = std_vals ./ sqrt(R)

    return mean_vals, std_vals, sem_vals, num_records
end

function plot_all_observables()
    data_dir = "../../data/"

    # Define the list of configurations: (W, H, Target Energy)
    lattice_configs = [(100, 100, 100)]

    s = 41
    tt = 600
    eqt_str = "500.000"
    dt_str = "0.010"
    of = 1
    R = 100
    dt = parse(Float64, dt_str)
    eqt = parse(Float64, eqt_str)

    eq_step = floor(Int, eqt / dt)
    eq_step_adj = div(eq_step, of) * of
    eq_idx_c = div(eq_step_adj, of)
    eq_idx = eq_idx_c + 1

    # List of all newly computed signals and their plot labels
    signals = [
        ("acf_C_E", "Energy ACF C_E(t)"),
        ("acf_C_v", "Velocity ACF C_v(t)"),
        ("acf_C_SS", "Stretch-Stretch C_SS(t)"),
        ("acf_C_SB", "Stretch-Angle C_SB(t)"),
        ("acf_C_BS", "Angle-Stretch C_BS(t)"),
        ("acf_C_BB", "Angle-Angle C_BB(t)"),
        ("G_SS", "G_SS = <(L S)^2>"),
        ("G_SB", "G_SB = <(L S)(L B)>"),
        ("G_BB", "G_BB = <(L B)^2>")
    ]

    # Dynamically scale figure height based on the number of signals to plot them below one another
    fig = Figure(size=(1600, 400 * length(signals)), fontsize=23)
    lwidth = 5.0
    colors = Makie.wong_colors()

    axes = []
    for (i, (sig_name, sig_label)) in enumerate(signals)
        # Using linear scale because cross-correlations and covariances might cross zero
        ax = Axis(fig[i, 1], xlabel="Time (Post-Equilibration)", ylabel=sig_label)
        push!(axes, ax)
    end

    for (idx, (w, h, energy)) in enumerate(lattice_configs)
        c = colors[mod1(idx, length(colors))]
        e_str = @sprintf("%.2f", energy)
        lbl = "$(w)x$(h), E=$(energy)"

        strEnd = "_w-$(w)_h-$(h)_H-$(e_str)_s-$(s)_tt-$(tt)_eqt-$(eqt_str)_dt-$(dt_str)_r-$(R)_of-$(of).bin"

        for (i, (sig_name, _)) in enumerate(signals)
            file_path = joinpath(data_dir, sig_name * strEnd)

            if isfile(file_path)
                mean_val, _, _, num_records = process_observable(file_path, R)

                t_phy = (0:num_records-1) .* (dt * of)
                t_plot = t_phy[eq_idx:end] .- t_phy[eq_idx]

                lines!(axes[i], t_plot, mean_val[eq_idx:end], color=c, linewidth=lwidth, label=lbl)
            else
                println("Warning: File missing for signal $(sig_name) -> $(file_path)")
            end
        end
    end

    for ax in axes
        axislegend(ax, position=:rt)
    end

    save("../../img/ACFs_Individual_Signals.png", fig)
end

plot_all_observables()
