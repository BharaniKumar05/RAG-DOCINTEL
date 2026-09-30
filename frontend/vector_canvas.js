/**
 * Vector Space 2D Canvas Visualizer
 * Renders document chunks in 2D projection space with query beacon and citation links.
 */

class VectorSpaceCanvas {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');

    this.nodes = [];
    this.queryPoint = null;
    this.activeCitations = [];

    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isDragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;

    this.hoveredNode = null;
    this.pulsePhase = 0;

    // Document Color Map
    this.docColors = [
      '#06b6d4', // Cyan
      '#6366f1', // Indigo
      '#a855f7', // Purple
      '#10b981', // Emerald
      '#f59e0b', // Amber
      '#f43f5e', // Rose
      '#38bdf8', // Sky
      '#ec4899'  // Pink
    ];
    this.docColorMap = {};

    this._initEvents();
    this._startAnimationLoop();
  }

  resize() {
    if (!this.canvas) return;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width * window.devicePixelRatio;
    this.canvas.height = rect.height * window.devicePixelRatio;
    this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    this.width = rect.width;
    this.height = rect.height;
  }

  setNodes(nodes) {
    this.nodes = nodes || [];
    // Assign color per doc_id
    let colorIdx = 0;
    this.nodes.forEach(n => {
      if (!this.docColorMap[n.doc_id]) {
        this.docColorMap[n.doc_id] = this.docColors[colorIdx % this.docColors.length];
        colorIdx++;
      }
    });
  }

  setQueryAndCitations(queryCoords, citations) {
    this.queryPoint = queryCoords || null;
    this.activeCitations = citations || [];
  }

  _initEvents() {
    window.addEventListener('resize', () => this.resize());
    setTimeout(() => this.resize(), 100);

    // Mouse Drag Pan
    this.canvas.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.dragStartX = e.clientX - this.panX;
      this.dragStartY = e.clientY - this.panY;
    });

    window.addEventListener('mousemove', (e) => {
      if (this.isDragging) {
        this.panX = e.clientX - this.dragStartX;
        this.panY = e.clientY - this.dragStartY;
      } else {
        this._checkHover(e);
      }
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    // Zoom
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      this.zoom = Math.min(Math.max(0.4, this.zoom * zoomFactor), 3.0);
    });
  }

  _checkHover(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const centerX = (this.width / 2) + this.panX;
    const centerY = (this.height / 2) + this.panY;

    this.hoveredNode = null;

    for (let n of this.nodes) {
      const nodeScreenX = centerX + (n.x * 1.6 * this.zoom);
      const nodeScreenY = centerY + (n.y * 1.6 * this.zoom);
      const dist = Math.hypot(mouseX - nodeScreenX, mouseY - nodeScreenY);
      if (dist < 10) {
        this.hoveredNode = n;
        this.canvas.style.cursor = 'pointer';
        return;
      }
    }
    this.canvas.style.cursor = 'default';
  }

  _startAnimationLoop() {
    const animate = () => {
      this.pulsePhase = (this.pulsePhase + 0.04) % (Math.PI * 2);
      this.render();
      requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }

  render() {
    if (!this.ctx || !this.width || !this.height) return;

    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    // Draw grid background
    this._drawGrid();

    const centerX = (this.width / 2) + this.panX;
    const centerY = (this.height / 2) + this.panY;

    // Draw connection lines from Query Point to Active Citations
    if (this.queryPoint && this.activeCitations.length > 0) {
      const qScreenX = centerX + (this.queryPoint[0] * 1.6 * this.zoom);
      const qScreenY = centerY + (this.queryPoint[1] * 1.6 * this.zoom);

      for (let cit of this.activeCitations) {
        const coords = cit.coords_2d || [0, 0];
        const citScreenX = centerX + (coords[0] * 1.6 * this.zoom);
        const citScreenY = centerY + (coords[1] * 1.6 * this.zoom);

        // Gradient line
        const grad = ctx.createLinearGradient(qScreenX, qScreenY, citScreenX, citScreenY);
        grad.addColorStop(0, 'rgba(244, 63, 94, 0.7)');
        grad.addColorStop(1, 'rgba(6, 182, 212, 0.7)');

        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.moveTo(qScreenX, qScreenY);
        ctx.lineTo(citScreenX, citScreenY);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // Draw Chunk Nodes
    for (let node of this.nodes) {
      const nodeX = centerX + (node.x * 1.6 * this.zoom);
      const nodeY = centerY + (node.y * 1.6 * this.zoom);

      const color = this.docColorMap[node.doc_id] || '#06b6d4';
      const isRetrieved = this.activeCitations.some(c => c.chunk_id === node.chunk_id);

      ctx.beginPath();
      const radius = isRetrieved ? 7 : 4.5;
      ctx.arc(nodeX, nodeY, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = isRetrieved ? 16 : 4;
      ctx.fill();
      ctx.shadowBlur = 0;

      if (isRetrieved) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.8;
        ctx.stroke();
      }
    }

    // Draw Query Point with animated radar pulse
    if (this.queryPoint) {
      const qX = centerX + (this.queryPoint[0] * 1.6 * this.zoom);
      const qY = centerY + (this.queryPoint[1] * 1.6 * this.zoom);

      // Radar Pulse Rings
      const pulseRadius = 10 + (Math.sin(this.pulsePhase) + 1) * 6;
      ctx.beginPath();
      ctx.arc(qX, qY, pulseRadius, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.45)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Query Node Center
      ctx.beginPath();
      ctx.arc(qX, qY, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#f43f5e';
      ctx.shadowColor = '#f43f5e';
      ctx.shadowBlur = 14;
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Label
      ctx.fillStyle = '#fecdd3';
      ctx.font = '10px JetBrains Mono';
      ctx.fillText('Active Query', qX + 10, qY + 3);
    }

    // Draw Hover Tooltip
    if (this.hoveredNode) {
      this._drawTooltip(this.hoveredNode, centerX, centerY);
    }
  }

  _drawGrid() {
    const ctx = this.ctx;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;
    const step = 30 * this.zoom;
    
    for (let x = (this.panX % step); x < this.width; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, this.height);
      ctx.stroke();
    }
    for (let y = (this.panY % step); y < this.height; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(this.width, y);
      ctx.stroke();
    }
  }

  _drawTooltip(node, centerX, centerY) {
    const ctx = this.ctx;
    const x = centerX + (node.x * 1.6 * this.zoom) + 12;
    const y = centerY + (node.y * 1.6 * this.zoom) - 10;

    const title = node.doc_title.slice(0, 30);
    const sec = node.section_title.slice(0, 25);

    ctx.fillStyle = 'rgba(15, 22, 36, 0.92)';
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.5)';
    ctx.lineWidth = 1;

    const boxW = 200;
    const boxH = 50;
    ctx.fillRect(x, y, boxW, boxH);
    ctx.strokeRect(x, y, boxW, boxH);

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 10px Outfit';
    ctx.fillText(title, x + 8, y + 16);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '9px Outfit';
    ctx.fillText(`Sec: ${sec}`, x + 8, y + 30);
    ctx.fillText(`ID: ${node.chunk_id}`, x + 8, y + 42);
  }
}
