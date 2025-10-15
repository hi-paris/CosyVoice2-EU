# CosyVoice2 Fine-tuning Demo

This demo showcases research on component-wise fine-tuning strategies for adapting CosyVoice2 to French TTS.

## Features

### Interactive Architecture Diagram
- SVG-based interactive diagram showing the CosyVoice2 pipeline
- Clickable components with detailed information panels
- Visual indicators for fine-tuning status

### Progressive Audio Comparison
- Dynamic audio generation based on selected fine-tuning components
- Quality indicators showing improvements for each configuration
- Side-by-side comparison with baseline model

### Comprehensive Evaluation Results
- Multiple metrics: MOS scores, accent strength, WER, speaker similarity
- Data efficiency analysis showing quality vs training data amount
- Interactive charts using Chart.js

### Key Design Decisions

#### Architecture Visualization
- Single interactive SVG diagram rather than multiple component images
- Real-time visual feedback through fine-tuning indicators
- Hover and click states for intuitive interaction

#### Audio Management
- Automatic audio file mapping based on component combinations
- Graceful error handling for missing audio files
- Pause-others-when-playing functionality

#### Responsive Design
- Mobile-first responsive layout
- Consistent with existing SSML demo design system
- Accessible keyboard navigation and focus states

## File Structure

```
cosyvoice2-demo/
├── index.html          # Main demo page
├── style.css           # Styling (extends SSML demo design)
├── script.js           # Interactive functionality
├── README.md           # This file
└── audio/              # Audio samples directory
    ├── baseline.wav
    ├── llm_only.wav
    ├── flow_only.wav
    ├── hifigan_only.wav
    ├── llm_flow.wav
    ├── llm_hifigan.wav
    ├── flow_hifigan.wav
    └── all_finetuned.wav
```

## Audio File Requirements

The demo expects the following audio files (replace placeholder files with actual recordings):

1. **baseline.wav** - Original CosyVoice2 without fine-tuning
2. **llm_only.wav** - Only LLM component fine-tuned
3. **flow_only.wav** - Only Flow Matching component fine-tuned
4. **hifigan_only.wav** - Only HiFiGAN component fine-tuned
5. **llm_flow.wav** - LLM + Flow components fine-tuned
6. **llm_hifigan.wav** - LLM + HiFiGAN components fine-tuned
7. **flow_hifigan.wav** - Flow + HiFiGAN components fine-tuned
8. **all_finetuned.wav** - All components fine-tuned

All audio files should use the same French text sample for fair comparison.

## Sample Text

The demo uses this French text sample:
> "Bonjour, je m'appelle Marie et je travaille dans une entreprise de technologie à Paris. Aujourd'hui, nous allons explorer les capacités de synthèse vocale en français avec CosyVoice2."

## Customization

### Updating Evaluation Data
Edit the `sampleData` object in `script.js` to update chart data:

```javascript
const sampleData = {
    configurations: [...],
    mosScores: [...],
    accentStrength: [...],
    werScores: [...],
    speakerSimilarity: [...]
};
```

### Adding New Components
1. Add new component to the SVG architecture diagram
2. Update the `componentInfo` object in script.js
3. Add corresponding audio files
4. Update the `audioConfigurations` object

### Styling Changes
The demo uses the same design system as the SSML demo. Main color scheme:
- Primary: #667eea (purple-blue)
- Secondary: #764ba2 (purple)
- Warning: #f6ad55 (orange - for "in progress" status)
- Success: #48bb78 (green)
- Error: #e53e3e (red)

## Browser Compatibility

- Modern browsers with CSS Grid and Flexbox support
- SVG support required for architecture diagram
- HTML5 audio support required for audio playback
- Chart.js for data visualization

## Accessibility Features

- Keyboard navigation for all interactive elements
- ARIA labels and semantic HTML structure
- Focus indicators for visual feedback
- Reduced motion support for accessibility preferences
- Alt text and descriptive content for screen readers
