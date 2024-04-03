#!/bin/bash
input_dir=$1
ouput_dir=$2
samplerate=$3
geom_file=$4

# Run bandpass filter stage of Mountainsort
ml-run-process ephys.bandpass_filter \
    --inputs timeseries:$input_dir/converted_data.mda \
    --outputs timeseries_out:$ouput_dir/filt.mda \
    --parameters samplerate:$samplerate freq_min:250 freq_max:5000

ml-run-process ephys.whiten \
    --inputs timeseries:$ouput_dir/filt.mda \
    --outputs timeseries_out:$ouput_dir/pre1.mda.prv

ml-run-process ephys.mask_out_artifacts \
    --inputs timeseries:$ouput_dir/pre1.mda.prv \
    --outputs timeseries_out:$ouput_dir/pre.mda.prv

# Spike sorting
# Specify the detect threshold in standard deviations
ml-run-process ms4alg.sort \
    --inputs \
        timeseries:$ouput_dir/pre.mda.prv \
        geom:$geom_file \
    --outputs \
        firings_out:$ouput_dir/firings.mda \
    --parameters \
        detect_sign:0 \
        adjacency_radius:100 \
        detect_threshold:4.0 \
        clip_size:100 \

# Compute cluster metrics
ml-run-process ephys.compute_cluster_metrics \
    --inputs \
        timeseries:$ouput_dir/pre.mda.prv firings:$ouput_dir/firings.mda \
    --outputs \
        metrics_out:$ouput_dir/cluster_metrics.json \
    --parameters \
        samplerate:$samplerate \
        clip_size:100 \
        refrac_msec:2 \

# Compute templates from just filtered data (preferred since we get amplitudes in uV)
ml-run-process ephys.compute_templates \
    --inputs \
        timeseries:$ouput_dir/filt.mda firings:$ouput_dir/firings.mda \
    --outputs \
        templates_out:$ouput_dir/templates.mda \
    --parameters \
        clip_size:100

# Some more cluster metrics
ml-run-process ms3.isolation_metrics \
    --inputs \
        timeseries:$ouput_dir/pre.mda.prv firings:$ouput_dir/firings.mda \
    --outputs \
        metrics_out:$ouput_dir/isolation_metrics_out.json \
        pair_metrics_out:$ouput_dir/pair_metrics_out.json \
    --parameters \
        compute_bursting_parents:true

# combine metrics
ml-run-process ms3.combine_cluster_metrics \
    --inputs \
        metrics_list:$ouput_dir/cluster_metrics.json \
        metrics_list:$ouput_dir/isolation_metrics_out.json \
    --outputs \
        metrics_out:$ouput_dir/combine_metrics_new.json \

# create label map
ml-run-process ms4alg.create_label_map \
    --inputs \
        metrics:$ouput_dir/combine_metrics_new.json \
    --outputs \
        label_map_out:$ouput_dir/label_map_out.prv \

# apply label map
ml-run-process ms4alg.apply_label_map \
    --inputs \
        firings:$ouput_dir/firings.mda \
        label_map:$ouput_dir/label_map_out.prv \
    --outputs \
        firings_out:$ouput_dir/firings_curated.mda \
