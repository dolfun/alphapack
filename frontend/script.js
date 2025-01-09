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

    const gridContainer = document.getElementById('grid');
    gridContainer.innerHTML = '';

    const cellSize = 40;
    const cellMargin = 3;
    gridContainer.style.width = `${info.length * (cellSize + 2 * cellMargin)}px`;

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
        gridContainer.appendChild(cell);

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
    this.apply(i, j, length, width, (cell) => {
      if (this.isValidPlacement(i, j, length, width, height)) {
        cell.style.backgroundColor = '#ccffcc';
      } else {
        cell.style.backgroundColor = '#ffcccc';
      }
      cell.textContent = cell.value + height;
    });
  }

  removeHover(i, j, length, width) {
    this.apply(i, j, length, width, (cell) => {
      cell.style.backgroundColor = this.getCellColor(cell);
      cell.textContent = cell.value;
    });
  }
}

function createPackages(info) {
  const rng = new RNG(67666);

  const packages = [];
  for (let i = 0; i < info.nrPackages; ++i) {
    packages.push({
      length: rng.randint(2, 5),
      width : rng.randint(2, 5),
      height: rng.randint(2, 5),
      isPlaced: false
    });
  }
  return packages;
}

function renderPackages(packages) {
  const list = document.getElementById('packagesList');
  list.innerHTML = '';

  packages.forEach((package, index) => {
    if (index >= 10) return;

    const item = document.createElement('div');
    item.className = 'list-item';
    item.textContent = `${package.length}x${package.width}x${package.height}`;
    if (index === 0) item.style.fontWeight = 'bold';

    list.appendChild(item);
  });
}

function updateHeadings(info, packages) {
  const containerHeading = document.getElementById('containerHeading');
  const listHeading = document.getElementById('listHeading');

  let placedCount = 0, placedVolume = 0;
  packages.forEach((package, _) => {
    if (!package.isPlaced) return;
    ++placedCount;

    placedVolume += package.length * package.width * package.height;
  });

  let packingEfficiency = 100 * placedVolume / (info.length * info.length * info.height);
  containerHeading.innerText = `Packing Efficiency: ${packingEfficiency.toFixed(2)}%`;
  listHeading.innerText = `\n\nPackages Used: ${placedCount}`;
}

function updateInferenceInfo(grid, packages) {
  const gridContainer = document.getElementById('grid');
  const inferenceInfo = document.getElementById('inferenceInfo');
  inferenceInfo.textContent = `Value: ${gridContainer.inferenceData.value.toFixed(2)}`

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

    const { length, width, height } = packages[0];
    let priors_sum = 0;
    const priors = gridContainer.inferenceData.priors;
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
        const prior = gridContainer.inferenceData.priors[index];

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

function addEventListeners(info, grid, packages) {
  for (let i = 0; i < grid.info.length; ++i) {
    for (let j = 0; j < grid.info.length; ++j) {
      const cell = grid.cells[i][j];
      cell.addEventListener('click', () => {
        if (grid.inferenceMode || packages[0].isPlaced) return;

        const topPackage = packages[0];
        const { length, width, height } = topPackage;
        if (!grid.isValidPlacement(i, j, length, width, height)) return;

        grid.removeHover(0, 0, grid.length, grid.length);
        grid.apply(i, j, length, width, (cell) => {
          cell.value += height;
          cell.textContent = cell.value;
          cell.style.backgroundColor = grid.getCellColor(cell);
        });

        packages.shift();
        topPackage.isPlaced = true;
        packages.push(topPackage);

        getInferenceData(grid, packages);
        renderPackages(packages);
        updateHeadings(info, packages);

        cell.dispatchEvent(new MouseEvent('mouseenter'));
      });

      cell.addEventListener('mouseenter', () => {
        if (grid.inferenceMode || packages[0].isPlaced) return;
        const { length, width, height } = packages[0];
        grid.applyHover(i, j, length, width, height);
      });

      cell.addEventListener('mouseleave', () => {
        if (grid.inferenceMode || packages[0].isPlaced) return;
        const { length, width, _ } = packages[0];
        grid.removeHover(i, j, length, width);
      });
    }
  }
}

function getInferenceData(grid, packages) {
  const gridContainer = document.getElementById('grid');
  gridContainer.inferenceData = null;

  const gridData = [];
  for (let i = 0; i < grid.info.length; ++i) {
    for (let j = 0; j < grid.info.length; ++j) {
      gridData.push(grid.cells[i][j].value);
    }
  }

  const data = {
    'height_map': gridData,
    'packages': packages
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
    gridContainer.inferenceData = data;
    updateInferenceInfo(grid, packages);
  })
  .catch(error => {
    alert(error);
  })
}

function main(info) {
  const grid = new Grid(info);
  const packages = createPackages(info);
  renderPackages(packages);
  updateHeadings(info, packages);
  addEventListeners(info, grid, packages);

  getInferenceData(grid, packages);
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
