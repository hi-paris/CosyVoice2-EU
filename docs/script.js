// Smooth scrolling navigation
document.addEventListener('DOMContentLoaded', function() {
    console.log('CosyVoice2 European Languages Demo loaded successfully!');

    // Current language state
    let currentLanguage = 'fr';

    // Chart instances for metric selector
    let radarChartInstance = null;
    let learningCurveChartInstance = null;
    let mixMonoChartInstance = null;
    let baselineChartInstance = null;
    let efficiencyChartInstance = null;
    
    // Initialize all functionality
    initializeNavigation();
    initializeLanguageSelector();
    initializeMetricSelector();
    initializeArchitectureDiagram();
    initializeFinetuningControls();
    initializeAudioPlayers();
    initializeCharts();
    initializeScrollAnimations();
    initializeInteractiveResultsTable();

    function initializeNavigation() {
        // Navigation link functionality
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.section');

    // Update active navigation link based on scroll position
    function updateActiveNavLink() {
        let current = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            
            if (pageYOffset >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    }

    // Smooth scrolling for navigation links
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                const headerHeight = document.querySelector('.nav').offsetHeight;
                const targetPosition = targetSection.offsetTop - headerHeight - 20;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Update active link on scroll
    window.addEventListener('scroll', updateActiveNavLink);
    
    // Initial active link update
    updateActiveNavLink();
    }

    // Language selector functionality
    function initializeLanguageSelector() {
        // Handle radio buttons in interactive demo
        const languageRadios = document.querySelectorAll('input[name="language"]');
        const evalLanguageSelect = document.getElementById('eval-language-select');
        
        // Set initial language from checked radio button
        const checkedRadio = document.querySelector('input[name="language"]:checked');
        if (checkedRadio) {
            currentLanguage = checkedRadio.value;
        }
        
        // Set evaluation language selector to match
        if (evalLanguageSelect) {
            evalLanguageSelect.value = currentLanguage;
        }
        
        // Add listeners to radio buttons
        languageRadios.forEach(radio => {
            radio.addEventListener('change', function() {
                if (this.checked) {
                    currentLanguage = this.value;
                    
                    // Update evaluation language selector
                    if (evalLanguageSelect) {
                        evalLanguageSelect.value = currentLanguage;
                    }
                    
                    updateAudioSources();
                    updateAudioConfiguration();
                    updateSampleText();
                    refreshChartsForLanguage();
                }
            });
        });

        // Add listener to evaluation language selector
        if (evalLanguageSelect) {
            evalLanguageSelect.addEventListener('change', function() {
                currentLanguage = this.value;
                
                // Update radio buttons
                const targetRadio = document.querySelector(`input[name="language"][value="${currentLanguage}"]`);
                if (targetRadio) {
                    targetRadio.checked = true;
                }
                
                updateAudioSources();
                updateAudioConfiguration();
                updateSampleText();
                refreshChartsForLanguage();
            });
        }

        function updateSampleText() {
            const sampleTextElement = document.getElementById('sample-text');
            const promptLanguageElement = document.getElementById('prompt-language-inline');
            
            const sampleTexts = {
                fr: '"Bonjour, je m\'appelle Luka et je travaille dans une entreprise de technologie à Paris. Aujourd\'hui, nous allons explorer les capacités de synthèse vocale en français avec CosyVoice 2."',
                de: '"Guten Tag, mein Name ist Hans und ich arbeite in einem Technologieunternehmen in Berlin. Heute werden wir die Sprachsynthesefähigkeiten von CosyVoice 2 im Deutschen erkunden."'
            };
            
            const languageNames = {
                fr: 'French',
                de: 'German'
            };
            
            if (sampleTextElement) {
                sampleTextElement.textContent = sampleTexts[currentLanguage];
            }
            
            if (promptLanguageElement) {
                promptLanguageElement.textContent = languageNames[currentLanguage];
            }
        }

        function updateAudioSources() {
            // Update prompt audio files for language change
            const promptFiles = {
                fr: 'common_voice_fr_40952142.wav',
                de: 'common_voice_de_41123857.wav'
            };

            // Update all audio sources based on language
            const promptSource = document.getElementById('prompt-source');
            if (promptSource) {
                promptSource.src = promptFiles[currentLanguage];
                const audio = promptSource.parentElement;
                audio.load();
            }
        }
        
        // Initialize
        updateSampleText();
        updateAudioSources();
    }

    // Metric selector functionality for charts
    function initializeMetricSelector() {
    const metricSelect = document.getElementById('metric-select');
        const evalLanguageSelect = document.getElementById('eval-language-select');
        if (!metricSelect) return;
        
    let currentMetric = metricSelect.value || 'wer_norm';
        let evalLanguage = evalLanguageSelect ? evalLanguageSelect.value || 'fr' : currentLanguage;
        
        // Set evaluation language to match current language initially
        if (evalLanguageSelect) {
            evalLanguageSelect.value = currentLanguage;
            evalLanguage = currentLanguage;
        }
        
        metricSelect.addEventListener('change', function() {
            currentMetric = this.value;
            loadAndRenderCharts(evalLanguage, currentMetric);
        });

        // Add evaluation language selector listener
        if (evalLanguageSelect) {
            evalLanguageSelect.addEventListener('change', function() {
                evalLanguage = this.value;
                loadAndRenderCharts(evalLanguage, currentMetric);
            });
        }

        // Load chart data and render charts
        function loadAndRenderCharts(language, metric) {
            console.log(`Loading charts for language: ${language}, metric: ${metric}`);
                loadRadarChart(language, metric);
                loadLearningCurveChart(language, metric);
                loadMixMonoCurveChart(language, metric);
                loadBaselineChart(language, metric);
        }

        // Radar chart logic (per-metric)
        function loadRadarChart(language, metric) {
            fetch(`generated_charts/radar_${language}_${metric}.json`)
                .then(response => {
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    return response.json();
                })
                .then(data => {
                    if (radarChartInstance) radarChartInstance.destroy();
                    const ctx = document.getElementById('radar-chart');
                    if (ctx) {
                        radarChartInstance = new Chart(ctx.getContext('2d'), {
                            type: 'radar',
                            data: data,
                            options: {
                                plugins: {
                                    tooltip: {
                                        callbacks: {
                                            label: function(context) {
                                                const value = context.parsed.r;
                                                return `${context.label}: ${value}`;
                                            }
                                        }
                                    },
                                    title: {
                                        display: true,
                                        text: `${data.metric_label || metric.toUpperCase()} Comparison (Absolute Values)`,
                                        font: { size: 18 }
                                    },
                                    legend: { position: 'top' }
                                },
                                scales: {
                                    r: {
                                        beginAtZero: false,
                                        min: data.min || 0,
                                        max: data.max || 100,
                                        pointLabels: { font: { size: 14 } }
                                    }
                                }
                            }
                        });
                    }
                })
                .catch(error => {
                    console.log('Radar chart data not available:', error);
                    const ctx = document.getElementById('radar-chart');
                    if (ctx) {
                        const parent = ctx.parentElement;
                        parent.innerHTML = `<p style="text-align:center; color:#666; padding:40px;">Chart data for ${language}/${metric} not yet available</p>`;
                    }
                });
        }

        // Learning curve chart logic (per-metric)
    function loadLearningCurveChart(language, metric) {
            fetch(`generated_charts/learning_curve_${language}_${metric}.json`)
                .then(response => {
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    return response.json();
                })
                .then(data => {
                    if (learningCurveChartInstance) learningCurveChartInstance.destroy();
                    const ctx = document.getElementById('learning-curve-chart');
                    if (ctx) {
                        learningCurveChartInstance = new Chart(ctx.getContext('2d'), {
                            type: 'line',
                            data: data,
                            options: {
                                plugins: {
                                    title: {
                                        display: true,
                                        text: `${data.metric_label || metric.toUpperCase()} Learning Curve`,
                                        font: { size: 18 }
                                    },
                                    legend: { position: 'top' }
                                },
                                scales: {
                                    x: { title: { display: true, text: data.x_label || 'Epoch' } },
                                    y: { title: { display: true, text: data.metric_label || metric.toUpperCase() }, beginAtZero: true }
                                }
                            }
                        });
                    }
                })
                .catch(error => {
                    console.log('Learning curve chart data not available:', error);
                    const ctx = document.getElementById('learning-curve-chart');
                    if (ctx) {
                        const parent = ctx.parentElement;
                        parent.innerHTML = `<p style="text-align:center; color:#666; padding:40px;">Chart data for ${language}/${metric} not yet available</p>`;
                    }
                });
        }

        // Mix vs Mono chart logic (per-metric)
        function loadMixMonoCurveChart(language, metric) {
            fetch(`generated_charts/mix_mono_curve_${language}_${metric}.json`)
                .then(response => {
                    if (!response.ok) throw new Error(`Failed to load mix mono chart: ${response.status}`);
                    return response.json();
                })
                .then(data => {
                    const ctx = document.getElementById('mix-mono-curve-chart');
                    if (!ctx) return;
                    
                    if (mixMonoChartInstance) {
                        mixMonoChartInstance.destroy();
                    }
                    
                    mixMonoChartInstance = new Chart(ctx, {
                        type: 'line',
                        data: data,
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                title: {
                                    display: true,
                                    text: `Mix vs Mono Training (${data.metric_label})`
                                },
                                legend: {
                                    position: 'top'
                                }
                            },
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: data.x_label || 'Training Hours'
                                    }
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: data.metric_label
                                    }
                                }
                            }
                        }
                    });
                })
                .catch(error => {
                    console.error('Error loading mix mono chart:', error);
                    const ctx = document.getElementById('mix-mono-curve-chart');
                    if (ctx) {
                        const parent = ctx.parentElement;
                        parent.innerHTML = '<p style="text-align: center; color: #666;">Chart data not available</p>';
                    }
                });
        }

        // Add fullscreen functionality
        function addFullscreenButton(chartContainer, chartId) {
            const fullscreenBtn = document.createElement('button');
            fullscreenBtn.className = 'fullscreen-btn';
            fullscreenBtn.innerHTML = '⛶';
            fullscreenBtn.title = 'Fullscreen';
            
            chartContainer.style.position = 'relative';
            chartContainer.appendChild(fullscreenBtn);
            
            fullscreenBtn.addEventListener('click', function() {
                const originalChart = Chart.getChart(chartId);
                if (!originalChart) {
                    console.error('No chart found with ID:', chartId);
                    return;
                }

                const modal = document.createElement('div');
                modal.style.cssText = `
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100vw;
                    height: 100vh;
                    background: rgba(0,0,0,0.9);
                    z-index: 9999;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                    box-sizing: border-box;
                `;
                
                const modalContent = document.createElement('div');
                modalContent.style.cssText = `
                    background: white;
                    border-radius: 10px;
                    padding: 30px;
                    width: 95%;
                    height: 90%;
                    position: relative;
                    display: flex;
                    flex-direction: column;
                `;
                
                const closeBtn = document.createElement('button');
                closeBtn.innerHTML = '×';
                closeBtn.style.cssText = `
                    position: absolute;
                    top: 15px;
                    right: 20px;
                    background: none;
                    border: none;
                    font-size: 30px;
                    cursor: pointer;
                    color: #666;
                    font-weight: bold;
                    z-index: 10;
                `;
                
                const chartTitle = document.createElement('h3');
                chartTitle.style.cssText = `
                    margin: 0 0 20px 0;
                    color: #333;
                    text-align: center;
                `;
                chartTitle.textContent = originalChart.options.plugins?.title?.text || 'Chart';
                
                const canvasContainer = document.createElement('div');
                canvasContainer.style.cssText = `
                    flex: 1;
                    position: relative;
                    min-height: 0;
                `;
                
                const canvas = document.createElement('canvas');
                canvas.id = chartId + '-fullscreen';
                canvas.style.cssText = `
                    width: 100% !important;
                    height: 100% !important;
                `;
                
                canvasContainer.appendChild(canvas);
                modalContent.appendChild(closeBtn);
                modalContent.appendChild(chartTitle);
                modalContent.appendChild(canvasContainer);
                modal.appendChild(modalContent);
                document.body.appendChild(modal);
                
                // Clone chart in fullscreen with proper sizing
                setTimeout(() => {
                    const fullscreenChart = new Chart(canvas, {
                        type: originalChart.config.type,
                        data: JSON.parse(JSON.stringify(originalChart.data)), // Deep clone data
                        options: {
                            ...JSON.parse(JSON.stringify(originalChart.options)), // Deep clone options
                            responsive: true,
                            maintainAspectRatio: false,
                            devicePixelRatio: window.devicePixelRatio || 1
                        }
                    });
                }, 100);
                
                // Close functionality
                function closeModal() {
                    const fullscreenChart = Chart.getChart(chartId + '-fullscreen');
                    if (fullscreenChart) {
                        fullscreenChart.destroy();
                    }
                    document.body.removeChild(modal);
                }
                
                closeBtn.addEventListener('click', closeModal);
                modal.addEventListener('click', function(e) {
                    if (e.target === modal) closeModal();
                });
                
                // Add escape key listener
                const escapeHandler = function(e) {
                    if (e.key === 'Escape') {
                        closeModal();
                        document.removeEventListener('keydown', escapeHandler);
                    }
                };
                document.addEventListener('keydown', escapeHandler);
            });
        }

        // Fullscreen button logic for all charts
        document.querySelectorAll('.chart-container').forEach(container => {
            const canvas = container.querySelector('canvas');
            if (canvas) {
                addFullscreenButton(container, canvas.id);
            }
        });

        // Baseline chart logic (per-metric)
        function loadBaselineChart(language, metric) {
            fetch(`generated_charts/baseline_${language}_${metric}.json`)
                .then(response => {
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    return response.json();
                })
                .then(data => {
                    if (baselineChartInstance) baselineChartInstance.destroy();
                    const ctx = document.getElementById('baseline-chart');
                    if (ctx) {
                        baselineChartInstance = new Chart(ctx.getContext('2d'), {
                            type: 'bar',
                            data: data,
                            options: {
                                responsive: true,
                                maintainAspectRatio: false,
                                devicePixelRatio: window.devicePixelRatio || 2,
                                plugins: {
                                    title: {
                                        display: true,
                                        text: `${data.metric_label || metric.toUpperCase()}: Baseline Comparison`,
                                        font: { size: 18 }
                                    },
                                    legend: { position: 'top' }
                                },
                                scales: {
                                    x: { title: { display: true, text: 'Model' } },
                                    y: { title: { display: true, text: data.metric_label || metric.toUpperCase() }, beginAtZero: true }
                                }
                            }
                        });
                    }
                })
                .catch(error => {
                    console.log('Baseline chart data not available:', error);
                    const ctx = document.getElementById('baseline-chart');
                    if (ctx) {
                        const parent = ctx.parentElement;
                        parent.innerHTML = `<p style="text-align:center; color:#666; padding:40px;">Chart data for ${language}/${metric} not yet available</p>`;
                    }
                });
        }

        // Efficiency chart logic (per-metric)
        function loadEfficiencyChart(language) {
            fetch(`generated_charts/efficiency_dual_${language}.json`)
                .then(response => {
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    return response.json();
                })
                .then(data => {
                    if (efficiencyChartInstance) efficiencyChartInstance.destroy();
                    const ctx = document.getElementById('efficiency-chart');
                    if (ctx) {
                        efficiencyChartInstance = new Chart(ctx.getContext('2d'), {
                            type: 'line',
                            data: { labels: data.labels, datasets: data.datasets },
                            options: {
                                plugins: {
                                    title: {
                                        display: true,
                                        text: `Data Efficiency: WER (norm) and SECS vs Training Hours`,
                                        font: { size: 18 }
                                    },
                                    legend: { position: 'top' }
                                },
                                scales: {
                                    x: { title: { display: true, text: 'Training Hours' } },
                                    yWER: { type: 'linear', position: 'left', title: { display: true, text: 'WER (normalized)' } },
                                    ySECS: { type: 'linear', position: 'right', title: { display: true, text: 'SECS' }, grid: { drawOnChartArea: false } }
                                }
                            }
                        });
                    }
                })
                .catch(error => {
                    console.log('Efficiency chart data not available:', error);
                    const ctx = document.getElementById('efficiency-chart');
                    if (ctx) {
                        const parent = ctx.parentElement;
                        parent.innerHTML = `<p style="text-align:center; color:#666; padding:40px;">Efficiency chart data for ${language} not yet available</p>`;
                    }
                });
        }

        // Initial load
    loadAndRenderCharts(evalLanguage, currentMetric);
    }

    function initializeArchitectureDiagram() {
        // Interactive architecture diagram functionality
        const componentBoxes = document.querySelectorAll('.component-box');
        const finetuneindicators = document.querySelectorAll('.finetune-indicator');
        const infoContent = document.getElementById('info-content');
        
        const componentInfo = {
            'slm': {
                name: 'Text-Speech Language Model',
                description: 'Converts input text into semantic tokens that represent the linguistic content. This component handles language-specific patterns and phonetic understanding.',
                impact: 'Fine-tuning this component improves pronunciation accuracy and language-specific prosody patterns for French and German.'
            },
            'flow': {
                name: 'Flow Matching Decoder (CFM)', 
                description: 'Uses conditional flow matching (non-autoregressive, streaming-capable) to map semantic tokens + prompt conditioning (speaker embedding, reference mel, prompt semantics) into mel spectrograms.',
                impact: 'Fine-tuning refines prosody, timing and continuity (e.g., French nasal vowels, German consonant clusters) after LM adaptation handles linguistic content.'
            },
            'hifigan': {
                name: 'HiFi-GAN Vocoder',
                description: 'Converts mel-spectrograms into high-quality waveforms, generating the final audio output with natural prosody and timbre.',
                impact: 'Fine-tuning improves audio quality and reduces artifacts specific to European language phonemes.'
            }
        };
        
        componentBoxes.forEach(box => {
            box.addEventListener('click', function() {
                // Remove active class from all components
                componentBoxes.forEach(b => b.classList.remove('active'));
                finetuneindicators.forEach(i => i.classList.remove('active'));
                
                // Add active class to clicked component
                this.classList.add('active');
                const componentType = this.getAttribute('data-component');
                const indicator = document.querySelector(`.${componentType}-indicator`);
                if (indicator) indicator.classList.add('active');
                
                // Update info panel
                const info = componentInfo[componentType];
                if (info && infoContent) {
                    infoContent.innerHTML = `
                        <h3>${info.name}</h3>
                        <p><strong>Function:</strong> ${info.description}</p>
                        <p><strong>Fine-tuning Impact:</strong> ${info.impact}</p>
                    `;
                }
            });
            
            // Add hover effects
            box.addEventListener('mouseenter', function() {
                if (!this.classList.contains('active')) {
                    this.style.opacity = '0.8';
                }
            });
            
            box.addEventListener('mouseleave', function() {
                this.style.opacity = '1';
            });
        });
        
        console.log('Architecture diagram initialized');
    }

    function initializeFinetuningControls() {
        // Fine-tuning controls functionality - use existing HTML elements
        const toggles = document.querySelectorAll('.component-toggle');
        
        if (toggles.length === 0) {
            console.warn('No component toggles found');
            return;
        }
        
        // Add event listeners to existing toggles
        toggles.forEach(toggle => {
            toggle.addEventListener('change', function() {
                updateAudioConfiguration();
                updateArchitectureVisualization();
                updateResultsTableHighlight();
            });
        });
        
        // Initialize with default state
        updateAudioConfiguration();
        updateArchitectureVisualization();
    updateResultsTableHighlight();
        
        console.log('Fine-tuning controls initialized');
    }

    function initializeCharts() {
        // Charts initialization placeholder
        console.log('Charts initialized');
    }

    function refreshChartsForLanguage() {
        // Refresh charts when language changes
        const metricSelect = document.getElementById('metric-select');
        const evalLanguageSelect = document.getElementById('eval-language-select');
    const currentMetric = metricSelect ? metricSelect.value : 'wer_norm';
        
        // Update evaluation language selector to match current language
        if (evalLanguageSelect) {
            evalLanguageSelect.value = currentLanguage;
        }
        
        // Destroy existing charts
        if (radarChartInstance) radarChartInstance.destroy();
        if (learningCurveChartInstance) learningCurveChartInstance.destroy();
        if (mixMonoChartInstance) mixMonoChartInstance.destroy();
        if (baselineChartInstance) baselineChartInstance.destroy();
        if (efficiencyChartInstance) efficiencyChartInstance.destroy();
        
        // Reload with new language
        setTimeout(() => {
            initializeMetricSelector();
        }, 100);

    // Update results table language emphasis
    updateResultsTableHeaderForLanguage();
    updateResultsTableHighlight();
    }

    function updateAudioConfiguration() {
        // Update audio based on fine-tuning configuration (restored from original)
        const dynamicTitle = document.getElementById('dynamic-title');
        const dynamicIndicators = document.getElementById('dynamic-indicators');
        const dynamicDescription = document.getElementById('dynamic-description');
        const dynamicAudio = document.getElementById('dynamic-audio');
        const dynamicSource = document.getElementById('dynamic-source');
        
        if (!dynamicTitle) return;
        
        const toggles = document.querySelectorAll('.component-toggle');
        
        // Audio configurations for different combinations (restored from original)
        const audioConfigurations = {
            'baseline': {
                file: 'original',
                title: 'Baseline (No Fine-tuning)',
                indicators: ['Strong English Accent', 'Unnatural Prosody'],
                indicatorClasses: ['poor', 'poor'],
                description: 'Original CosyVoice2 model without any language-specific training.'
            },
            'slm': {
                file: 'slm',
                title: 'Text-Speech LM Fine-tuned',
                indicators: ['Improved Pronunciation', 'Some English Accent'],
                indicatorClasses: ['good', 'neutral'],
                description: 'Text-Speech Language Model fine-tuned for language-specific pronunciation patterns.'
            },
            'flow': {
                file: 'flow',
                title: 'Flow Fine-tuned',
                indicators: ['Better Prosody', 'Pronunciation Issues'],
                indicatorClasses: ['good', 'neutral'],
                description: 'Flow matching model fine-tuned for natural prosody and rhythm.'
            },
            'hifigan': {
                file: 'hifigan',
                title: 'HiFi-GAN Optimized* <span style="font-size: 0.8em;">(=Original CosyVoice2 model)</span>',
                indicators: ['Cleaner Audio', 'Accent Remains'],
                indicatorClasses: ['good', 'poor'],
                description: 'Using the official CosyVoice2 HiFi-GAN model instead of partially trained version.'
            },
            'flow_slm': {
                file: 'slm_flow',
                title: 'Text-Speech LM + Flow Fine-tuned',
                indicators: ['Good Pronunciation', 'Natural Prosody'],
                indicatorClasses: ['good', 'good'],
                description: 'Combined Text-Speech LM and Flow fine-tuning for improved pronunciation and prosody.'
            },
            'hifigan_slm': {
                file: 'slm_hifigan',
                title: 'Text-Speech LM + HiFiGAN Optimized*',
                indicators: ['Good Pronunciation', 'Clean Audio'],
                indicatorClasses: ['good', 'good'],
                description: 'Combined Text-Speech LM fine-tuning with official CosyVoice2 HiFiGAN model.'
            },
            'flow_hifigan': {
                file: 'flow_hifigan',
                title: 'Flow + HiFiGAN Optimized*',
                indicators: ['Natural Prosody', 'Clean Audio'],
                indicatorClasses: ['good', 'good'],
                description: 'Combined Flow fine-tuning with official CosyVoice2 HiFiGAN model.'
            },
            'flow_hifigan_slm': {
                file: 'slm_flow_hifigan',
                title: 'All Components Optimized*',
                indicators: ['Excellent Pronunciation', 'Natural Prosody', 'High Quality Audio'],
                indicatorClasses: ['good', 'good', 'good'],
                description: 'Text-Speech LM and Flow fine-tuned with official CosyVoice2 HiFiGAN for optimal quality.'
            }
        };
        
        const activeComponents = [];
        toggles.forEach(toggle => {
            if (toggle.checked) {
                activeComponents.push(toggle.getAttribute('data-component'));
            }
        });

        const configKey = activeComponents.length > 0 ? activeComponents.sort().join('_') : 'baseline';
        const config = audioConfigurations[configKey] || audioConfigurations['baseline'];

        // Update dynamic audio card
        dynamicTitle.innerHTML = `<i class="fas fa-magic"></i> ${config.title}`;
        
        // Update quality indicators
        if (dynamicIndicators) {
            dynamicIndicators.innerHTML = config.indicators.map((indicator, index) => 
                `<span class="quality-badge ${config.indicatorClasses[index]}">${indicator}</span>`
            ).join('');
        }

        // Update audio source with language suffix
        if (dynamicSource && dynamicAudio) {
            const audioFile = `audio/${config.file}-${currentLanguage}.wav`;
            dynamicSource.src = audioFile;
            dynamicAudio.load();
        }

        // Update description
        if (dynamicDescription) {
            dynamicDescription.textContent = config.description;
        }

        console.log(`Updated configuration to: ${configKey} for language: ${currentLanguage}`);
    }

    function updateArchitectureVisualization() {
        // Update architecture diagram based on fine-tuning selection (restored from original)
        const toggles = document.querySelectorAll('.component-toggle');
        const indicators = document.querySelectorAll('.finetune-indicator');
        
        // Reset all indicators in the architecture diagram
        indicators.forEach(indicator => {
            indicator.classList.remove('active');
            indicator.classList.add('inactive');
        });

        // Activate indicators for checked components
        toggles.forEach(toggle => {
            if (toggle.checked) {
                const component = toggle.getAttribute('data-component');
                const indicator = document.querySelector(`.${component}-indicator`);
                if (indicator) {
                    indicator.classList.remove('inactive');
                    indicator.classList.add('active');
                }
            }
        });
        
        console.log('Architecture visualization updated');
    }

    // --- Interactive Results Table ---
    function initializeInteractiveResultsTable() {
        const tableEl = document.getElementById('config-table');
        if (!tableEl) return;

        // Data derived from the provided LaTeX table (rounded to two decimals)
        // Each row corresponds to a (slm, flow, hifigan) combination with metrics per regime
        // Symbols: circle=original (o), plus=fine-tuned (+), minus=partially trained (-)
        const rows = [
            { key:'o_o_-', slm:'o', flow:'o', hifigan:'-', fr:{wer:53.16,secs:0.23,mcd:8.39}, de:{wer:66.45,secs:0.19,mcd:8.12}, bili:{wer:'52.74/73.49', secs:'0.23/0.19', mcd:'8.44/8.09'}, note:'baseline' },
            { key:'o_o_o', slm:'o', flow:'o', hifigan:'o', fr:{wer:51.53,secs:0.15,mcd:9.67}, de:{wer:66.77,secs:0.14,mcd:8.85}, bili:{wer:'49.59/68.44', secs:'0.15/0.13', mcd:'9.68/8.84'}, shaded:true },
            { key:'o_+_-', slm:'o', flow:'+', hifigan:'-', fr:{wer:51.72,secs:0.24,mcd:8.63}, de:{wer:65.16,secs:0.21,mcd:8.22}, bili:{wer:'51.62/65.02', secs:'0.26/0.24', mcd:'8.50/8.38'} },
            { key:'o_+_o', slm:'o', flow:'+', hifigan:'o', fr:{wer:49.67,secs:0.16,mcd:9.93}, de:{wer:62.97,secs:0.18,mcd:8.98}, bili:{wer:'52.98/62.36', secs:'0.24/0.22', mcd:'9.61/9.26'} },
            { key:'+_+_-', slm:'+', flow:'+', hifigan:'-', fr:{wer:9.67,secs:0.31,mcd:7.76}, de:{wer:6.56,secs:0.27,mcd:7.39}, bili:{wer:'10.18/8.11', secs:'0.33/0.30', mcd:'7.58/7.68'}, best:true },
            { key:'+_o_-', slm:'+', flow:'o', hifigan:'-', fr:{wer:10.48,secs:0.30,mcd:7.80}, de:{wer:7.41,secs:0.25,mcd:7.44}, bili:{wer:'10.85/7.17', secs:'0.29/0.26', mcd:'7.72/7.39'} },
            { key:'+_o_o', slm:'+', flow:'o', hifigan:'o', fr:{wer:9.76,secs:0.24,mcd:8.60}, de:{wer:7.28,secs:0.21,mcd:7.80}, bili:{wer:'10.45/7.38', secs:'0.22/0.22', mcd:'8.53/7.80'} },
            { key:'+_+_o', slm:'+', flow:'+', hifigan:'o', fr:{wer:8.77,secs:0.24,mcd:8.58}, de:{wer:6.03,secs:0.25,mcd:7.75}, bili:{wer:'9.03/6.42', secs:'0.31/0.28', mcd:'8.28/8.08'}, shaded:true, final:true }
        ];

        // Render header
        const headerHtml = `
            <thead>
                <tr>
                    <th colspan="3" class="subhead">Configuration</th>
                    <th colspan="3" class="subhead mono-head">FR (mono)</th>
                    <th colspan="3" class="subhead mono-head">DE (mono)</th>
                    <th colspan="3" class="subhead bili-head">FR+DE (bilingual; FR/DE)</th>
                </tr>
                <tr>
                    <th>LLM</th><th>Flow</th><th>Voc.</th>
                    <th>WER↓</th><th>SECS↑</th><th>MCD↓</th>
                    <th>WER↓</th><th>SECS↑</th><th>MCD↓</th>
                    <th>WER↓</th><th>SECS↑</th><th>MCD↓</th>
                </tr>
            </thead>
        `;

        // Render body
        const bodyHtml = rows.map(r => `
            <tr data-key="${r.key}">
                <td>${symbol(r.slm)}</td>
                <td>${symbol(r.flow)}</td>
                <td>${symbol(r.hifigan)}</td>
                <td>${fmt(r.fr.wer)}</td>
                <td>${fmt(r.fr.secs)}</td>
                <td>${fmt(r.fr.mcd)}</td>
                <td>${fmt(r.de.wer)}</td>
                <td>${fmt(r.de.secs)}</td>
                <td>${fmt(r.de.mcd)}</td>
                <td>${r.bili.wer}</td>
                <td>${r.bili.secs}</td>
                <td>${r.bili.mcd}</td>
            </tr>
        `).join('');

        tableEl.innerHTML = headerHtml + `<tbody>${bodyHtml}</tbody>`;

        updateResultsTableHeaderForLanguage();
        updateResultsTableHighlight();

        function symbol(s) {
            if (s === '+') return '⊕';
            if (s === '-') return '⊖';
            return '◦';
        }
        function fmt(v) {
            return typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(2)) : v;
        }
    }

    function updateResultsTableHeaderForLanguage() {
        const tableEl = document.getElementById('config-table');
        if (!tableEl) return;
        const isFR = currentLanguage === 'fr';
        // Dim non-selected mono block slightly
        const thRows = tableEl.querySelectorAll('thead tr');
        if (thRows.length >= 2) {
            // Reset
            tableEl.querySelectorAll('thead th').forEach(th => th.style.opacity = '1');
            // Apply emphasis
            const monoStart = isFR ? 3 : 6;
            const monoEnd = isFR ? 5 : 8;
            const otherStart = isFR ? 6 : 3;
            const otherEnd = isFR ? 8 : 5;
            // Slightly fade the non-selected mono language columns
            for (let i = otherStart; i <= otherEnd; i++) {
                thRows[1].children[i].style.opacity = '0.6';
            }
        }
    }

    function updateResultsTableHighlight() {
        const tableEl = document.getElementById('config-table');
        if (!tableEl) return;
        // Determine selected components
        const slm = document.getElementById('slm-toggle')?.checked ? '+' : 'o';
        const flow = document.getElementById('flow-toggle')?.checked ? '+' : 'o';
        // For HiFi-GAN, treat checked as 'o' (optimized official) vs unchecked as '-' (partially trained) to match data rows
        const hifigan = document.getElementById('hifigan-toggle')?.checked ? 'o' : '-';
        const key = `${slm}_${flow}_${hifigan}`;
        tableEl.querySelectorAll('tbody tr').forEach(tr => tr.classList.remove('highlight'));
        const match = tableEl.querySelector(`tbody tr[data-key="${key}"]`);
        if (match) {
            match.classList.add('highlight');
            // Scroll into view if table is off-screen a bit
            // match.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
    }

    function initializeScrollAnimations() {
        // Scroll animations placeholder
        console.log('Scroll animations initialized');
    }

    function initializeAudioPlayers() {
        // Audio player enhancements
    const audioElements = document.querySelectorAll('audio');
    
    audioElements.forEach(audio => {
        // Add loading state
        audio.addEventListener('loadstart', function() {
            this.parentElement.classList.add('loading');
        });
        
        audio.addEventListener('canplaythrough', function() {
            this.parentElement.classList.remove('loading');
        });
        
        // Pause other audio when one starts playing
        audio.addEventListener('play', function() {
            audioElements.forEach(otherAudio => {
                if (otherAudio !== audio) {
                    otherAudio.pause();
                }
            });
        });

        // Add keyboard accessibility
        audio.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                if (this.paused) {
                    this.play();
                } else {
                    this.pause();
                }
            }
        });
    });

    // Demo section animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe demo pairs for animation
    const demoPairs = document.querySelectorAll('.demo-pair');
    demoPairs.forEach((pair, index) => {
        pair.style.opacity = '0';
        pair.style.transform = 'translateY(30px)';
        pair.style.transition = `opacity 0.6s ease ${index * 0.1}s, transform 0.6s ease ${index * 0.1}s`;
        observer.observe(pair);
    });

    // Observe overview cards for animation
    const overviewCards = document.querySelectorAll('.overview-card');
    overviewCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = `opacity 0.6s ease ${index * 0.2}s, transform 0.6s ease ${index * 0.2}s`;
        observer.observe(card);
    });

    // Audio comparison highlighting
    const audioItems = document.querySelectorAll('.audio-item');
    audioItems.forEach(item => {
        const audio = item.querySelector('audio');
        
        audio.addEventListener('play', function() {
            item.classList.add('playing');
            // Add subtle glow effect
            item.style.boxShadow = '0 0 20px rgba(102, 126, 234, 0.3)';
        });
        
        audio.addEventListener('pause', function() {
            item.classList.remove('playing');
            item.style.boxShadow = '';
        });
        
        audio.addEventListener('ended', function() {
            item.classList.remove('playing');
            item.style.boxShadow = '';
        });
    });

    // Error handling for audio files
    audioElements.forEach(audio => {
        audio.addEventListener('error', function() {
            const errorMsg = document.createElement('p');
            errorMsg.textContent = 'Audio file not found. Please upload the audio file.';
            errorMsg.style.color = '#e53e3e';
            errorMsg.style.fontSize = '0.9rem';
            errorMsg.style.fontStyle = 'italic';
            
            this.parentElement.appendChild(errorMsg);
            this.style.display = 'none';
        });
    });

    // Header background scroll effect
    const header = document.querySelector('.header');
    const nav = document.querySelector('.nav');
    
    window.addEventListener('scroll', function() {
        const scrolled = window.pageYOffset;
        const rate = scrolled * -0.5;
        
        if (header) {
            header.style.transform = `translateY(${rate}px)`;
        }
        
        // Add background to nav when scrolling
        if (scrolled > 100) {
            nav.style.background = 'rgba(255, 255, 255, 0.95)';
            nav.style.backdropFilter = 'blur(10px)';
        } else {
            nav.style.background = 'rgba(255, 255, 255, 0.9)';
            nav.style.backdropFilter = 'blur(10px)';
        }
    });

    // Copy link functionality for sharing
    function createShareButton() {
        const shareBtn = document.createElement('button');
        shareBtn.innerHTML = '<i class="fas fa-share"></i> Share Demo';
        shareBtn.className = 'share-btn';
        shareBtn.style.cssText = `
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 20px;
            border-radius: 50px;
            cursor: pointer;
            font-weight: 500;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            z-index: 1000;
        `;
        
        shareBtn.addEventListener('click', function() {
            if (navigator.share) {
                navigator.share({
                    title: 'CosyVoice2-EU: FR/DE Generative TTS Demo',
                    text: 'Interactive CosyVoice2 adaptation demo for expressive French & German zero-shot TTS',
                    url: window.location.href
                });
            } else {
                // Fallback: copy URL to clipboard
                navigator.clipboard.writeText(window.location.href).then(() => {
                    shareBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                    setTimeout(() => {
                        shareBtn.innerHTML = '<i class="fas fa-share"></i> Share Demo';
                    }, 2000);
                });
            }
        });
        
        shareBtn.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.boxShadow = '0 6px 25px rgba(0, 0, 0, 0.3)';
        });
        
        shareBtn.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.2)';
        });
        
        document.body.appendChild(shareBtn);
    }

    // Create share button
    createShareButton();

    // Performance optimization: Lazy load audio files
    const audioObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const audio = entry.target;
                const sources = audio.querySelectorAll('source');
                sources.forEach(source => {
                    if (source.dataset.src) {
                        source.src = source.dataset.src;
                        source.removeAttribute('data-src');
                    }
                });
                audio.load();
                audioObserver.unobserve(audio);
            }
        });
    }, { rootMargin: '100px' });

    // Uncomment the following lines if you want to implement lazy loading for audio files
    // audioElements.forEach(audio => {
    //     audioObserver.observe(audio);
    // });
    }

    console.log('CosyVoice2 European Languages Demo loaded successfully!');
});
