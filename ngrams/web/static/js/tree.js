/**
 * D3.js Tree Visualization for N-gram Predictions
 * 
 * Creates an interactive radial tree showing word predictions
 * branching from a root word based on n-gram probabilities.
 */

// Color scale based on probability
const colorScale = d3.scaleSequential()
    .domain([0, 1])
    .interpolator(d3.interpolateViridis);

// Alternative color scale (warmer)
const warmColorScale = d3.scaleSequential()
    .domain([0, 1])
    .interpolator(d3.interpolateYlOrRd);

let svg, g, tooltip;
let currentData = null;

// Initialize the visualization
function initTree() {
    const container = d3.select('#tree-container');
    const width = container.node().clientWidth;
    const height = container.node().clientHeight;
    
    // Clear previous
    container.selectAll('*').remove();
    
    // Create SVG
    svg = container.append('svg')
        .attr('width', width)
        .attr('height', height);
    
    // Add zoom behavior
    const zoom = d3.zoom()
        .scaleExtent([0.3, 3])
        .on('zoom', (event) => {
            g.attr('transform', event.transform);
        });
    
    svg.call(zoom);
    
    // Main group for transformation
    g = svg.append('g')
        .attr('transform', `translate(${width/2}, ${height/2})`);
    
    // Tooltip
    tooltip = d3.select('#tooltip');
    
    // Add legend
    const legend = container.append('div')
        .attr('class', 'legend');
    
    legend.append('div')
        .attr('class', 'legend-gradient');
    
    legend.append('div')
        .html('Low ← Probability → High');
}

// Render the tree with data
function renderTree(data) {
    if (!data) return;
    
    currentData = data;
    
    const container = d3.select('#tree-container');
    const width = container.node().clientWidth;
    const height = container.node().clientHeight;
    const radius = Math.min(width, height) / 2 - 80;
    
    // Re-initialize if needed
    if (!svg) {
        initTree();
    }
    
    // Clear previous content
    g.selectAll('*').remove();
    
    // Create hierarchy
    const root = d3.hierarchy(data);
    
    // Create radial tree layout
    const treeLayout = d3.tree()
        .size([2 * Math.PI, radius])
        .separation((a, b) => (a.parent == b.parent ? 1 : 2) / a.depth);
    
    treeLayout(root);
    
    // Create links
    const link = g.selectAll('.link')
        .data(root.links())
        .join('path')
        .attr('class', 'link')
        .attr('d', d3.linkRadial()
            .angle(d => d.x)
            .radius(d => d.y))
        .attr('stroke', d => colorScale(d.target.data.probability || 0))
        .attr('stroke-width', d => Math.max(1, d.target.data.probability * 5))
        .attr('stroke-opacity', d => 0.3 + d.target.data.probability * 0.5);
    
    // Create node groups
    const node = g.selectAll('.node')
        .data(root.descendants())
        .join('g')
        .attr('class', d => 'node' + (d.children ? ' node--internal' : ' node--leaf'))
        .attr('transform', d => `
            rotate(${d.x * 180 / Math.PI - 90})
            translate(${d.y}, 0)
        `);
    
    // Add circles to nodes
    node.append('circle')
        .attr('r', d => {
            if (d.depth === 0) return 20;  // Root
            const baseSize = 8;
            const probSize = d.data.probability * 12;
            return baseSize + probSize;
        })
        .attr('fill', d => {
            if (d.depth === 0) return '#4dabf7';  // Root color
            return colorScale(d.data.probability || 0);
        })
        .attr('stroke', d => d.depth === 0 ? '#fff' : d3.rgb(colorScale(d.data.probability || 0)).darker())
        .on('mouseover', handleMouseOver)
        .on('mouseout', handleMouseOut)
        .on('click', handleClick);
    
    // Add labels
    node.append('text')
        .attr('dy', '0.31em')
        .attr('x', d => d.x < Math.PI === !d.children ? 6 : -6)
        .attr('text-anchor', d => d.x < Math.PI === !d.children ? 'start' : 'end')
        .attr('transform', d => d.x >= Math.PI ? 'rotate(180)' : null)
        .text(d => d.data.name)
        .attr('font-size', d => d.depth === 0 ? '14px' : '11px')
        .attr('font-weight', d => d.depth === 0 ? 'bold' : 'normal')
        .attr('fill', '#333')
        .clone(true).lower()
        .attr('stroke', '#fff')
        .attr('stroke-width', 3);
    
    // Add probability labels for non-root nodes
    node.filter(d => d.depth > 0)
        .append('text')
        .attr('dy', '-1em')
        .attr('x', 0)
        .attr('text-anchor', 'middle')
        .attr('transform', d => d.x >= Math.PI ? 'rotate(180)' : null)
        .text(d => (d.data.probability * 100).toFixed(1) + '%')
        .attr('font-size', '9px')
        .attr('fill', '#666');
}

// Mouse over handler
function handleMouseOver(event, d) {
    const [x, y] = d3.pointer(event, document.body);
    
    // Highlight node
    d3.select(this).classed('node-highlight', true);
    
    // Highlight path to root
    let node = d;
    while (node.parent) {
        g.selectAll('.link')
            .filter(link => link.target === node)
            .classed('link-highlight', true);
        node = node.parent;
    }
    
    // Show tooltip
    const probPct = (d.data.probability * 100).toFixed(2);
    
    tooltip.html(`
        <div class="word">${d.data.name}</div>
        <div class="prob">Probability: ${probPct}%</div>
        ${d.data.count ? `<div class="count">Count: ${d.data.count.toLocaleString()}</div>` : ''}
        ${d.depth > 0 ? `<div>Depth: ${d.depth}</div>` : ''}
    `)
    .style('left', (x + 15) + 'px')
    .style('top', (y - 10) + 'px')
    .classed('visible', true);
}

// Mouse out handler
function handleMouseOut(event, d) {
    // Remove highlight
    d3.select(this).classed('node-highlight', false);
    g.selectAll('.link').classed('link-highlight', false);
    
    // Hide tooltip
    tooltip.classed('visible', false);
}

// Click handler - make this word the new root
function handleClick(event, d) {
    if (d.depth === 0) return;  // Don't re-click root
    
    // Update tree with clicked word as root (updateTree is defined in index.html)
    if (typeof updateTree === 'function') {
        currentWord = d.data.name;
        updateTree(d.data.name, true);  // true = add to path
    }
}

// Handle window resize
window.addEventListener('resize', () => {
    if (currentData) {
        initTree();
        renderTree(currentData);
    }
});

// Initialize on load
document.addEventListener('DOMContentLoaded', initTree);
