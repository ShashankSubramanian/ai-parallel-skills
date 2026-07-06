# Shared Initialization

Use this subskill to synchronize replicated parameters after metadata has been
attached.

## Required Helper

Implement a helper equivalent to:

```python
def init_params_for_shared_weights(model):
    with torch.no_grad():
        for param in model.parameters():
            metadata = param.comm_metadata
            for comm_group in metadata["shared"]:
                if comm.get_size(comm_group) > 1:
                    dist.broadcast(
                        param,
                        src=comm.get_root(comm_group),
                        group=comm.get_group(comm_group),
                    )
```

## Rules

- Call this after model initialization and after all `comm_metadata` has been
  attached.
- Broadcast only across groups listed in `shared`.
- Do not broadcast parameters solely for ordinary `dp`; DDP handles data-parallel
  model copies.
- Do not modify sharded parameters except across groups where their local shard
  is replicated.
- Run validation before broadcasting so missing metadata fails clearly.

## Output

The target project should have one explicit initialization call site, for
example:

```python
annotate_model_parameters(model, plan)
validate_param_comm_metadata(model)
init_params_for_shared_weights(model)
```
