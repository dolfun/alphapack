class RNG {
  constructor(seed) {
    this.state = seed;
  }

  rand() {
    let x = this.state;
    x ^= x >> 17;
    x *= 0xed5ad4bb;
    x ^= x >> 11;
    x *= 0xac4c1b51;
    x ^= x >> 15;
    x *= 0x31848bab;
    x ^= x >> 14;
    this.state = x;
    return x;
  }

  randint(a, b) {
    return a + this.rand() % (b - a + 1);
  }
}

class Grid {
  constructor(info) {
    this.info = info;

    const gridElement = document.getElementById('grid');
    gridElement.innerHTML = '';

    const cellSize = 40;
    const cellMargin = 3;
    gridElement.style.width = `${info.length * (cellSize + 2 * cellMargin)}px`;

    const cells = [];
    for (let i = 0; i < info.length; ++i) {
      cells[i] = [];
      for (let j = 0; j < info.length; ++j) {
        const cell = document.createElement('div');
        cell.className = 'grid-cell';
        cell.style.width = `${cellSize}px`;
        cell.style.height = `${cellSize}px`;
        cell.style.margin = `${cellMargin}px`;
        cell.value = 0;
        cell.textContent = '0';
        cell.style.backgroundColor = this.getCellColor(cell);
        gridElement.appendChild(cell);

        cells[i][j] = cell;
      }
    }

    this.cells = cells;
    this.inferenceMode = false;
  }

  getCellColor(cell) {
    let value = cell.value / this.info.height;
    value = Math.round(214 - 255 / 2 * value);
    return `rgb(${value}, ${value}, ${value})`;
  }

  apply(i0, j0, length, width, func) {
    for (let i = i0; i < Math.min(i0 + length, this.info.length); ++i) {
      for (let j = j0; j < Math.min(j0 + width, this.info.length); ++j) {
        func(this.cells[i][j]);
      }
    }
  };

  isValidPlacement(i, j, length, width, height) {
    if (i + length > this.info.length || j + width > this.info.length) return false;
      
    let max_val = -1, max_count = 0;
    this.apply(i, j, length, width, (cell) => {
      if (cell.value > max_val) {
        max_val = cell.value;
        max_count = 1;
      } else if (cell.value == max_val) {
        ++max_count;
      }
    });
    if (max_val + height > this.info.height) return false;

    return true;
  };

  applyHover(i, j, length, width, height) {
    let maxHeight = -1;
    this.apply(i, j, length, width, (cell) => {
      maxHeight = Math.max(maxHeight, cell.value);
    });

    this.apply(i, j, length, width, (cell) => {
      if (this.isValidPlacement(i, j, length, width, height)) {
        cell.style.backgroundColor = '#ccffcc';
      } else {
        cell.style.backgroundColor = '#ffcccc';
      }
      cell.textContent = maxHeight + height;
    });
  }

  removeHover(i, j, length, width) {
    this.apply(i, j, length, width, (cell) => {
      cell.style.backgroundColor = this.getCellColor(cell);
      cell.textContent = cell.value;
    });
  }
}

function createitems(info) {
  const rng = new RNG(67666);

  const items = [];
  for (let i = 0; i < info.itemCount; ++i) {
    items.push({
      length: rng.randint(2, 5),
      width : rng.randint(2, 5),
      height: rng.randint(2, 5),
      isPlaced: false
    });
  }
  return items;
}

function renderitems(items) {
  const list = document.getElementById('itemsList');
  list.innerHTML = '';

  items.forEach((item, index) => {
    if (index >= 10) return;

    const div = document.createElement('div');
    div.className = 'list-item';
    div.textContent = `${item.length}x${item.width}x${item.height}`;
    if (index === 0) div.style.fontWeight = 'bold';

    list.appendChild(div);
  });
}

function updateHeadings(info, items) {
  const stateHeading = document.getElementById('stateHeading');
  const listHeading = document.getElementById('listHeading');

  let placedCount = 0, placedVolume = 0;
  items.forEach((item, _) => {
    if (!item.isPlaced) return;
    ++placedCount;

    placedVolume += item.length * item.width * item.height;
  });

  let packingEfficiency = 100 * placedVolume / (info.length * info.length * info.height);
  stateHeading.innerText = `Packing Efficiency: ${packingEfficiency.toFixed(2)}%`;
  listHeading.innerText = `\n\nitems Used: ${placedCount}`;
}

function updateInferenceInfo(grid, items) {
  const gridElement = document.getElementById('grid');
  const inferenceInfo = document.getElementById('inferenceInfo');
  inferenceInfo.textContent = `Value: ${gridElement.inferenceData.value.toFixed(2)}`

  const inferenceLabel = document.getElementById('inferenceLabel');
  if (inferenceLabel.children.length == 0) {
    const inferenceCheckboxText = document.createElement('span');
    inferenceCheckboxText.textContent = 'Policy: ';
    inferenceLabel.appendChild(inferenceCheckboxText);

    const inferenceCheckbox = document.createElement('input');
    inferenceCheckbox.type = 'checkbox';
    inferenceCheckbox.id = 'inferenceCheckbox';
    inferenceLabel.appendChild(inferenceCheckbox);
  }

  let inferenceCheckbox = document.getElementById('inferenceCheckbox');
  inferenceCheckbox.addEventListener('click', () => {
    if (inferenceCheckbox.checked) {
      grid.inferenceMode = true;
    } else {
      grid.inferenceMode = false;
    }

    const { length, width, height } = items[0];
    let priors_sum = 0;
    const priors = gridElement.inferenceData.priors;
    for (let i = 0; i < grid.info.length; ++i) {
      for (let j = 0; j < grid.info.length; ++j) {
        const index = i * grid.info.length + j;
        if (grid.isValidPlacement(i, j, length, width, height)) {
          priors_sum += priors[index];
        } else {
          priors[index] = 0;
        }
      }
    }

    for (let i = 0; i < priors.length; ++i) {
      priors[i] /= priors_sum;
    }

    for (let i = 0; i < grid.info.length; ++i) {
      for (let j = 0; j < grid.info.length; ++j) {
        const index = i * grid.info.length + j;
        const prior = gridElement.inferenceData.priors[index];

        let color = null;
        if (inferenceCheckbox.checked) {
          const val = Math.max(1 - 10 * prior, 0);
          const hue = 120 * val;
          const saturation = 50;
          const lightness = 60;
          color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;

        } else {
          color = grid.getCellColor(grid.cells[i][j]);
        }

        grid.cells[i][j].style.backgroundColor = color;
      }
    }
  });
}

function addEventListeners(info, grid, items) {
  for (let i = 0; i < grid.info.length; ++i) {
    for (let j = 0; j < grid.info.length; ++j) {
      const cell = grid.cells[i][j];
      cell.addEventListener('click', () => {
        if (grid.inferenceMode || items[0].isPlaced) return;

        const topitem = items[0];
        const { length, width, height } = topitem;
        if (!grid.isValidPlacement(i, j, length, width, height)) return;

        grid.removeHover(0, 0, grid.length, grid.length);

        let maxHeight = -1;
        grid.apply(i, j, length, width, (cell) => {
          maxHeight = Math.max(maxHeight, cell.value);
        });

        grid.apply(i, j, length, width, (cell) => {
          cell.value = maxHeight + height;
          if (cell.value > grid.info.height) alert('Max height exceeded!');

          cell.textContent = cell.value;
          cell.style.backgroundColor = grid.getCellColor(cell);
        });

        items.shift();
        topitem.isPlaced = true;
        items.push(topitem);

        getInferenceData(grid, items);
        renderitems(items);
        updateHeadings(info, items);

        cell.dispatchEvent(new MouseEvent('mouseenter'));
      });

      cell.addEventListener('mouseenter', () => {
        if (grid.inferenceMode || items[0].isPlaced) return;
        const { length, width, height } = items[0];
        grid.applyHover(i, j, length, width, height);
      });

      cell.addEventListener('mouseleave', () => {
        if (grid.inferenceMode || items[0].isPlaced) return;
        const { length, width, _ } = items[0];
        grid.removeHover(i, j, length, width);
      });
    }
  }
}

function getInferenceData(grid, items) {
  const gridElement = document.getElementById('grid');
  gridElement.inferenceData = null;

  const gridData = [];
  for (let i = 0; i < grid.info.length; ++i) {
    for (let j = 0; j < grid.info.length; ++j) {
      gridData.push(grid.cells[i][j].value);
    }
  }

  const data = {
    'height_map': gridData,
    'items': items
  };

  const ipInput = document.getElementById('ipInput');
  const url = 'http://' + ipInput.value + '/infer';
  fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('Invalid Response');
    }
    return response.json();
  })
  .then(data => {
    gridElement.inferenceData = data;
    updateInferenceInfo(grid, items);
  })
  .catch(error => {
    alert(error);
  })
}

function main(info) {
  const grid = new Grid(info);
  const items = createitems(info);
  renderitems(items);
  updateHeadings(info, items);
  addEventListeners(info, grid, items);
  getInferenceData(grid, items);
}

const connectButton = document.getElementById('connectButton');
connectButton.addEventListener('click', () => {
  const ipInput = document.getElementById('ipInput');
  const url = 'http://' + ipInput.value + '/info';
  fetch(url)
    .then(response => {
      if (!response.ok) {
        throw new Error('Invalid Response');
      }
      return response.json();
    })
    .then(info => {
    main(info);
    })
    .catch(error => {
      alert(error);
    })
});
