"""
TODO
diffusion evalatution over 30 timesteps for 32 samples

sampling strategies (solving time per generation for each)
1. DDPM (guidance)
2. DDIM (guidance, eta, num_steps)
3. DPM++ (guidance, num_steps)

metric evaluations
1. CPRS
2. Spread + skill
3. RMSE from ground (deterministic, 1st member, mean)

graphical evaluations
1. std of each channell at each timestep

replication
1. save the full history in marked torch files

extra:
1. using xN iterative walks


FLOW (saving full history)

1. get deterministic 
advance and save

2. diffusion
select sampler -> start time -> xN for iterative walks ->
end time when done -> log all metrics -> log graphics ->
save data for replication
"""