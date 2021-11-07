#IntelliStat

## Table of Contents

 * [Project structure](#project-structure)
 * [Implemented Examples](#implemented-examples)
 * [Code examples](#code-examples)

## Project structure

- [__IntelliStat__](IntelliStat) - project codebase
  - [__datasets__](IntelliStat/datasets)
  - [__generic\_builders__](IntelliStat/generic_builders)
    - [__component\_builder__](IntelliStat/generic_builders/component_builder) - component(part of shape) builder
    - [__model\_builder__](IntelliStat/generic_builders/model_builder) - builder of one of available neural network models
    - [__shape\_builder__](IntelliStat/generic_builders/shape_builder) - builder of one of specified shapes in [__shapes__](IntelliStat/generic_builders/shape_builder/shapes)
      - [__shapes__](IntelliStat/generic_builders/shape_builder/shapes) - library of hardcoded shapes as json filed
  - [__neural\_networks__](IntelliStat/neural_networks) - collection of implemented neural networks
- [__examples__](examples) - collection of IntelliStat code usage
- [__jupyter__](jupyter) - jupyter notebook scripts
- [README.md](README.md) - project description
- [requirements.txt](requirements.txt) - requirements for using IntelliStat

## Implemented Examples

[Documentation](examples/README.md) of implemented examples


## Code examples

### Build model based on config.json

``` python
from IntelliStat.generic_builders import ModelBuilder

# Create model builder object
builder = ModelBuilder()

# Specify configuration and validation file
config_schema = Path(__file__).parent / 'resources/config_schema.json'
config_file = Path(__file__).parent / 'resources/config.json'

# Build model
neural_network_model = builder.build_model(config_file=config_file, config_schema_file=config_schema)
```

### Get configuration data as types.SimpleNamespace object

``` python
from IntelliStat.generic_builders import ModelBuilder

# Create model builder object
builder = ModelBuilder()

# Specify configuration and validation file
config_schema = Path(__file__).parent / 'resources/config_schema.json'
config_file = Path(__file__).parent / 'resources/config.json'

# Retrieve configuration
configuration: SimpleNamespace = builder.load_configuration(config_file=config_file, config_schema_file=config_schema)

# Access configuration elements by dot notation
epoch: int = configuration.epoch
```

### Create shape using shape builder

```python
from pathlib import Path
import numpy as np
from IntelliStat.generic_builders import ShapeBuilder

# Create shape object - Gauss
shape = ShapeBuilder.Gauss

# Define X data
X_data = [[X / 2 for X in range(20)] for _ in range(50)]
X_data = np.array(X_data, dtype=np.float32)

# Generate Gauss shape based on IntelliStat/generic_builder/shape_builder/shapes/gauss.json
# custom Gauss configuration can be used
# shape.config_file = 'custom_gauss.json'
Y_data = shape.build_shape(X_data)
```

### Different ways to specify how to access shape builder
```python
from IntelliStat.generic_builders import ShapeBuilder

# shape_by_dot, shape_by_index, shape_by_name 
# are the same shape builder object

# dot notation
shape_by_dot = ShapeBuilder.Gauss

# by index
shape_by_index = ShapeBuilder[0]

# by name
shape_by_name = ShapeBuilder('Gauss')
```

### Different ways to specify how to access component builder
```python
from IntelliStat.generic_builders import ComponentBuilder

# component_by_dot, component_by_index, component_by_name 
# are the same component builder object

# dot notation
component_by_dot = ComponentBuilder.Gauss

# by index
component_by_index = ComponentBuilder[0]

# by name
component_by_name = ComponentBuilder('Gauss')
```