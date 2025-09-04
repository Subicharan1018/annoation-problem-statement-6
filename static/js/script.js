// Image Annotation App - Frontend JavaScript
// This contains dummy functionality for UI interaction

class AnnotationApp {
    constructor() {
        this.currentTool = 'select';
        this.uploadedImages = [];
        this.annotations = [];
        this.currentImageIndex = -1;
        
        this.initializeElements();
        this.bindEvents();
        
        console.log('🎨 Image Annotation App initialized');
    }
    
    initializeElements() {
        // Get DOM elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.canvas = document.getElementById('annotationCanvas');
        this.canvasPlaceholder = document.getElementById('canvasPlaceholder');
        this.imageList = document.getElementById('imageList');
        this.annotationList = document.getElementById('annotationList');
        
        // Tool buttons
        this.toolButtons = document.querySelectorAll('.tool-btn');
        this.selectTool = document.getElementById('selectTool');
        this.boxTool = document.getElementById('boxTool');
        this.polygonTool = document.getElementById('polygonTool');
        
        // Action buttons
        this.clearBtn = document.getElementById('clearBtn');
        this.saveBtn = document.getElementById('saveBtn');
        
        // Properties
        this.annotationLabel = document.getElementById('annotationLabel');
        this.annotationColor = document.getElementById('annotationColor');
        this.annotationOpacity = document.getElementById('annotationOpacity');
        
        // Canvas context
        this.ctx = this.canvas.getContext('2d');
    }
    
    bindEvents() {
        // File upload events
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        this.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Tool selection events
        this.toolButtons.forEach(btn => {
            btn.addEventListener('click', this.handleToolSelect.bind(this));
        });
        
        // Action button events
        this.clearBtn.addEventListener('click', this.clearAnnotations.bind(this));
        this.saveBtn.addEventListener('click', this.saveAnnotation.bind(this));
        
        // Canvas events (dummy for now)
        this.canvas.addEventListener('mousedown', this.handleCanvasMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleCanvasMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleCanvasMouseUp.bind(this));
        
        // Properties events
        this.annotationLabel.addEventListener('input', this.updateAnnotationProperties.bind(this));
        this.annotationColor.addEventListener('change', this.updateAnnotationProperties.bind(this));
        this.annotationOpacity.addEventListener('input', this.updateAnnotationProperties.bind(this));
    }
    
    // File handling methods
    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
        console.log('📁 File dragged over upload area');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }
    
    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files).filter(file => 
            file.type.startsWith('image/')
        );
        this.processFiles(files);
        console.log(`📁 ${files.length} files dropped`);
    }
    
    handleFileSelect(e) {
        const files = Array.from(e.target.files).filter(file => 
            file.type.startsWith('image/')
        );
        this.processFiles(files);
        console.log(`📁 ${files.length} files selected`);
    }
    
    processFiles(files) {
        files.forEach((file, index) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const imageData = {
                    id: Date.now() + index,
                    name: file.name,
                    size: file.size,
                    url: e.target.result,
                    annotations: []
                };
                
                this.uploadedImages.push(imageData);
                this.updateImageList();
                
                // Load first image automatically
                if (this.currentImageIndex === -1) {
                    this.loadImage(0);
                }
                
                console.log(`🖼️ Image loaded: ${file.name}`);
            };
            reader.readAsDataURL(file);
        });
        
        // Show success alert
        alert(`📁 Successfully uploaded ${files.length} image(s)!`);
    }
    
    updateImageList() {
        if (this.uploadedImages.length === 0) {
            this.imageList.innerHTML = '<p class="empty-state">No images uploaded yet</p>';
            return;
        }
        
        this.imageList.innerHTML = this.uploadedImages.map((image, index) => `
            <div class="image-item ${index === this.currentImageIndex ? 'active' : ''}" 
                 onclick="app.loadImage(${index})">
                <h4>${image.name}</h4>
                <p>${this.formatFileSize(image.size)} • ${image.annotations.length} annotations</p>
            </div>
        `).join('');
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }
    
    loadImage(index) {
        if (index < 0 || index >= this.uploadedImages.length) return;
        
        this.currentImageIndex = index;
        const image = this.uploadedImages[index];
        
        const img = new Image();
        img.onload = () => {
            // Resize canvas to fit image
            const maxWidth = 800;
            const maxHeight = 600;
            let { width, height } = img;
            
            if (width > maxWidth) {
                height = (height * maxWidth) / width;
                width = maxWidth;
            }
            if (height > maxHeight) {
                width = (width * maxHeight) / height;
                height = maxHeight;
            }
            
            this.canvas.width = width;
            this.canvas.height = height;
            
            // Draw image
            this.ctx.clearRect(0, 0, width, height);
            this.ctx.drawImage(img, 0, 0, width, height);
            
            // Show canvas and hide placeholder
            this.canvas.style.display = 'block';
            this.canvasPlaceholder.style.display = 'none';
            
            // Update UI
            this.updateImageList();
            this.updateAnnotationList();
        };
        img.src = image.url;
        
        console.log(`🖼️ Loaded image: ${image.name}`);
    }
    
    // Tool selection methods
    handleToolSelect(e) {
        const tool = e.currentTarget.dataset.tool;
        this.currentTool = tool;
        
        // Update active state
        this.toolButtons.forEach(btn => btn.classList.remove('active'));
        e.currentTarget.classList.add('active');
        
        console.log(`🛠️ Selected tool: ${tool}`);
        alert(`🛠️ Selected ${tool} tool`);
    }
    
    // Canvas interaction methods (dummy implementations)
    handleCanvasMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        console.log(`🖱️ Mouse down at (${x}, ${y}) with ${this.currentTool} tool`);
        
        if (this.currentTool === 'box') {
            alert(`📦 Started drawing bounding box at (${Math.round(x)}, ${Math.round(y)})`);
        } else if (this.currentTool === 'polygon') {
            alert(`🔺 Started drawing polygon at (${Math.round(x)}, ${Math.round(y)})`);
        }
    }
    
    handleCanvasMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Only log occasionally to avoid spam
        if (Math.random() < 0.01) {
            console.log(`🖱️ Mouse move: (${x}, ${y})`);
        }
    }
    
    handleCanvasMouseUp(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        console.log(`🖱️ Mouse up at (${x}, ${y})`);
        
        if (this.currentTool === 'box') {
            this.createDummyAnnotation('box', x, y);
        } else if (this.currentTool === 'polygon') {
            this.createDummyAnnotation('polygon', x, y);
        }
    }
    
    createDummyAnnotation(type, x, y) {
        if (this.currentImageIndex === -1) return;
        
        const annotation = {
            id: Date.now(),
            type: type,
            label: this.annotationLabel.value || `${type} annotation`,
            color: this.annotationColor.value,
            opacity: this.annotationOpacity.value,
            coordinates: { x: Math.round(x), y: Math.round(y) },
            timestamp: new Date().toISOString()
        };
        
        this.uploadedImages[this.currentImageIndex].annotations.push(annotation);
        this.annotations.push(annotation);
        
        this.updateAnnotationList();
        this.updateImageList();
        
        console.log(`✨ Created ${type} annotation:`, annotation);
        alert(`✨ Created ${type} annotation with label "${annotation.label}"`);
    }
    
    updateAnnotationList() {
        if (this.currentImageIndex === -1 || this.uploadedImages[this.currentImageIndex].annotations.length === 0) {
            this.annotationList.innerHTML = '<p class="empty-state">No annotations created yet</p>';
            return;
        }
        
        const annotations = this.uploadedImages[this.currentImageIndex].annotations;
        this.annotationList.innerHTML = annotations.map((annotation, index) => `
            <div class="annotation-item" onclick="app.selectAnnotation(${annotation.id})">
                <h4>${annotation.label}</h4>
                <p>${annotation.type} • (${annotation.coordinates.x}, ${annotation.coordinates.y})</p>
            </div>
        `).join('');
    }
    
    selectAnnotation(id) {
        const annotation = this.annotations.find(a => a.id === id);
        if (annotation) {
            console.log(`🎯 Selected annotation:`, annotation);
            alert(`🎯 Selected annotation: ${annotation.label}`);
        }
    }
    
    // Action methods
    clearAnnotations() {
        if (this.currentImageIndex === -1) {
            alert('⚠️ No image selected');
            return;
        }
        
        const confirmed = confirm('🗑️ Are you sure you want to clear all annotations for this image?');
        if (confirmed) {
            this.uploadedImages[this.currentImageIndex].annotations = [];
            this.annotations = this.annotations.filter(a => 
                !this.uploadedImages[this.currentImageIndex].annotations.includes(a)
            );
            
            this.updateAnnotationList();
            this.updateImageList();
            
            console.log('🗑️ Cleared all annotations');
            alert('🗑️ All annotations cleared');
        }
    }
    
    saveAnnotation() {
        if (this.currentImageIndex === -1) {
            alert('⚠️ No image selected');
            return;
        }
        
        const image = this.uploadedImages[this.currentImageIndex];
        const annotationData = {
            imageName: image.name,
            annotations: image.annotations,
            timestamp: new Date().toISOString()
        };
        
        console.log('💾 Saving annotation data:', annotationData);
        
        // In a real app, this would send data to the Flask backend
        // For now, just show success message
        alert(`💾 Saved ${image.annotations.length} annotation(s) for ${image.name}`);
    }
    
    updateAnnotationProperties() {
        console.log('🎨 Updated annotation properties:', {
            label: this.annotationLabel.value,
            color: this.annotationColor.value,
            opacity: this.annotationOpacity.value
        });
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AnnotationApp();
});

// Additional utility functions
function showNotification(message, type = 'info') {
    console.log(`📢 ${type.toUpperCase()}: ${message}`);
    // In a real app, this could show a toast notification
}

function exportAnnotations() {
    if (!window.app || window.app.uploadedImages.length === 0) {
        alert('⚠️ No annotations to export');
        return;
    }
    
    const exportData = window.app.uploadedImages.map(image => ({
        imageName: image.name,
        annotations: image.annotations
    }));
    
    console.log('📤 Exporting annotations:', exportData);
    alert('📤 Export functionality would download annotations as JSON');
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (!window.app) return;
    
    switch(e.key) {
        case '1':
            if (e.ctrlKey) {
                e.preventDefault();
                document.getElementById('selectTool').click();
            }
            break;
        case '2':
            if (e.ctrlKey) {
                e.preventDefault();
                document.getElementById('boxTool').click();
            }
            break;
        case '3':
            if (e.ctrlKey) {
                e.preventDefault();
                document.getElementById('polygonTool').click();
            }
            break;
        case 's':
            if (e.ctrlKey) {
                e.preventDefault();
                window.app.saveAnnotation();
            }
            break;
        case 'Delete':
            if (window.app.currentImageIndex !== -1) {
                window.app.clearAnnotations();
            }
            break;
    }
});

// Help function to show keyboard shortcuts
function showKeyboardShortcuts() {
    const shortcuts = [
        'Ctrl + 1: Select tool',
        'Ctrl + 2: Box tool', 
        'Ctrl + 3: Polygon tool',
        'Ctrl + S: Save annotations',
        'Delete: Clear annotations'
    ];
    
    alert('⌨️ Keyboard Shortcuts:\n\n' + shortcuts.join('\n'));
}

console.log('⌨️ Keyboard shortcuts enabled. Press Ctrl+1,2,3 for tools, Ctrl+S to save, Delete to clear.');
console.log('🚀 Image Annotation App ready! Upload images to get started.');