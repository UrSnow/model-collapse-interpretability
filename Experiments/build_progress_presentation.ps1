param(
    [string]$OutputPath = (Join-Path $PSScriptRoot 'presentations\project_progress_with_notes.pptx')
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$root = $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
    $OutputPath = Join-Path $root $OutputPath
}
$build = Join-Path $root 'presentations\.build'
New-Item -ItemType Directory -Force -Path $build | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $OutputPath) | Out-Null
$repo = 'https://github.com/UrSnow/model-collapse-interpretability'

function Rgb([string]$hex) {
    $hex = $hex.TrimStart('#')
    $r = [Convert]::ToInt32($hex.Substring(0, 2), 16)
    $g = [Convert]::ToInt32($hex.Substring(2, 2), 16)
    $b = [Convert]::ToInt32($hex.Substring(4, 2), 16)
    return $r + 256 * $g + 65536 * $b
}

$ink = Rgb '#20282B'
$muted = Rgb '#4B5D60'
$teal = Rgb '#147D78'
$orange = Rgb '#C75334'
$light = Rgb '#EDF2F1'
$white = Rgb '#FFFFFF'
$font = 'Aptos'
$codeFont = 'Consolas'

function Add-Text($slide, [string]$value, [double]$left, [double]$top,
                  [double]$width, [double]$height, [double]$size = 20,
                  [bool]$bold = $false, [int]$color = $ink,
                  [string]$family = $font) {
    $shape = $slide.Shapes.AddTextbox(1, $left, $top, $width, $height)
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = -1
    $range = $shape.TextFrame.TextRange
    $range.Text = $value
    $range.Font.Name = $family
    $range.Font.Size = $size
    $range.Font.Bold = $(if ($bold) { -1 } else { 0 })
    $range.Font.Color.RGB = $color
    return $shape
}

function Add-Notes($slide, [string]$value) {
    $placeholder = $slide.NotesPage.Shapes.Placeholders(2)
    $placeholder.TextFrame.TextRange.Text = $value
}

function New-Slide([string]$title, [string]$source, [string]$notes) {
    $notes = $notes.Replace("$repo/blob/main/", "$repo/blob/main/Experiments/")
    $notes = $notes.Replace("$repo/tree/main/", "$repo/tree/main/Experiments/")
    $notes = $notes.Replace("$repo/blob/main/Experiments/.gitignore", "$repo/blob/main/.gitignore")
    $notes = $notes.Replace("$repo/blob/main/Experiments/PROJECT_CHECKLIST.md", "$repo/blob/main/PROJECT_CHECKLIST.md")
    $slide = $deck.Slides.Add($deck.Slides.Count + 1, 12)
    $slide.Background.Fill.ForeColor.RGB = $white
    Add-Text $slide $title 48 27 850 54 30 $true $ink | Out-Null
    Add-Text $slide $source 48 506 810 22 11 $false $muted | Out-Null
    Add-Text $slide ([string]$deck.Slides.Count) 885 506 25 22 11 $false $muted | Out-Null
    Add-Notes $slide $notes
    return $slide
}

function Add-Line($slide, [string]$label, [string]$detail, [double]$top,
                  [double]$labelWidth = 235, [double]$fontSize = 18) {
    Add-Text $slide $label 52 $top $labelWidth 54 $fontSize $true $teal $codeFont | Out-Null
    Add-Text $slide $detail (58 + $labelWidth) $top (835 - $labelWidth) 54 $fontSize $false $ink | Out-Null
}

function Add-Figure($slide, [string]$file, [double]$left, [double]$top,
                    [double]$maxWidth, [double]$maxHeight) {
    if (-not (Test-Path -LiteralPath $file)) { throw "Missing figure: $file" }
    $bitmap = [System.Drawing.Image]::FromFile($file)
    try {
        $factor = [Math]::Min($maxWidth / $bitmap.Width, $maxHeight / $bitmap.Height)
        $width = $bitmap.Width * $factor
        $height = $bitmap.Height * $factor
    } finally { $bitmap.Dispose() }
    $x = $left + ($maxWidth - $width) / 2
    $y = $top + ($maxHeight - $height) / 2
    $slide.Shapes.AddPicture($file, 0, -1, $x, $y, $width, $height) | Out-Null
}

function Split-Figure([string]$source, [string]$stem) {
    $image = [System.Drawing.Bitmap]::new($source)
    try {
        $cut = [int][Math]::Floor($image.Height * 0.52)
        $topRect = [System.Drawing.Rectangle]::new(0, 0, $image.Width, $cut)
        $bottomRect = [System.Drawing.Rectangle]::new(0, $cut, $image.Width, ($image.Height - $cut))
        $topImage = $image.Clone($topRect, $image.PixelFormat)
        $bottomImage = $image.Clone($bottomRect, $image.PixelFormat)
        try {
            $topPath = Join-Path $build "$stem-top.png"
            $bottomPath = Join-Path $build "$stem-bottom.png"
            $topImage.Save($topPath, [System.Drawing.Imaging.ImageFormat]::Png)
            $bottomImage.Save($bottomPath, [System.Drawing.Imaging.ImageFormat]::Png)
        } finally {
            $topImage.Dispose()
            $bottomImage.Dispose()
        }
    } finally { $image.Dispose() }
    return @($topPath, $bottomPath)
}

$finalPanels = Split-Figure (Join-Path $root 'figures\final-evaluation\comparison.png') 'final'
$mechanismPanels = Split-Figure (Join-Path $root 'figures\interpret-extended\mechanistic_comparison.png') 'mechanism'

$ppt = $null
$deck = $null
try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $deck = $ppt.Presentations.Add(0)
    $deck.PageSetup.SlideWidth = 960
    $deck.PageSetup.SlideHeight = 540

    # 1. Title
    $slide = $deck.Slides.Add(1, 12)
    $slide.Background.Fill.ForeColor.RGB = $white
    Add-Text $slide 'Mechanistic interpretability of model collapse' 56 112 845 120 42 $true $ink | Out-Null
    Add-Text $slide 'Completed experiments and code walkthrough' 58 250 820 40 25 $false $teal | Out-Null
    Add-Text $slide '8 October 2026' 58 448 280 28 16 $false $muted | Out-Null
    Add-Text $slide 'UrSnow/model-collapse-interpretability' 58 478 780 24 15 $false $muted $codeFont | Out-Null
    Add-Notes $slide "This is a progress presentation, not the final course abstract. The implemented task is a discrete-token adaptation of the Omniglot-feature induction task. We have completed the experimental code and reports, but the submission abstract and final packaging remain. Source: $repo"

    # 2. Research question and scope
    $slide = New-Slide 'Research question and scope' 'README.txt; STEP2_REPORT.md' "Ask whether recursive training on a predecessor's predictions changes clean in-context learning and the mechanisms that support it. We analyze a base query-label task and an extended label, next-symbol, next-label task. The study uses abstract symbol tokens, not Omniglot image features; this affects how broadly the result can be interpreted. Sources: $repo/blob/main/README.txt and $repo/blob/main/STEP2_REPORT.md"
    Add-Text $slide 'Question' 54 112 170 32 22 $true $teal | Out-Null
    Add-Text $slide 'What changes when each new model trains on outputs from the previous one?' 54 147 830 78 25 $false $ink | Out-Null
    Add-Text $slide 'Measured outcomes' 54 266 245 32 22 $true $teal | Out-Null
    Add-Text $slide "Clean-task accuracy and loss`nNext-symbol distribution and label consistency`nAttention patterns and causal interventions" 54 305 830 150 22 $false $ink | Out-Null

    # 3. Task generator
    $slide = New-Slide 'Randomly relabelled symbol task' 'project1/data.py: SymbolTaskConfig, SymbolTaskGenerator.sample' "Each prompt has two distinct context symbols paired with two distinct labels. The query repeats one context symbol. The generator samples a new mapping on every sequence, so memorizing a global symbol-label dictionary cannot solve the task. The input sequence is s0 l0 s1 l1 query. This is implemented by SymbolTaskGenerator.sample in project1/data.py. Source: $repo/blob/main/project1/data.py"
    Add-Text $slide 's0  l0  s1  l1  query' 61 112 820 48 29 $true $ink $codeFont | Out-Null
    Add-Text $slide 'The query copies one context symbol; its target is that pair''s label.' 61 171 820 73 22 $false $ink | Out-Null
    Add-Text $slide "symbols = symbol_scores.topk(cfg.context_pairs, dim=1).indices`nlabels = label_scores.topk(cfg.context_pairs, dim=1).indices`ninput_ids[:, 0:-1:2] = symbols`ninput_ids[:, 1:-1:2] = labels + cfg.num_symbols" 61 280 840 156 18 $false $muted $codeFont | Out-Null

    # 4. Model and diagnostics
    $slide = New-Slide 'Transformer and induction diagnostics' 'project1/model.py; project1/diagnostics.py' "The model uses two causal transformer layers, eight attention heads per layer, residual width 64 and learned token and position embeddings. CausalSelfAttention applies a lower-triangular mask. The diagnostics module scores previous-token attention in layer one and query-to-matching-label attention in layer two; it also ablates heads. Attention scores describe patterns, while ablation tests functional contribution. Sources: $repo/blob/main/project1/model.py and $repo/blob/main/project1/diagnostics.py"
    Add-Text $slide 'Two layers, eight heads each, 64-dimensional residual stream' 58 110 830 61 24 $false $ink | Out-Null
    Add-Text $slide "scores = query @ key.transpose(-2, -1) / head_dim**0.5`nscores = scores.masked_fill(~causal_mask, -inf)`nhead_outputs = attention @ value" 58 219 840 132 19 $false $muted $codeFont | Out-Null
    Add-Text $slide 'Previous-token and induction scores are measured per head.' 58 386 830 58 22 $false $teal | Out-Null

    # 5. Baseline study
    $slide = New-Slide 'Step 1 baseline' 'project1/train.py; project1/sweep.py; project1/analyze.py' "The baseline trains each seed on a streaming randomly relabelled task. train.py handles optimization, validation checkpoint selection, locked test evaluation and attention logging. sweep.py launches seeds 0 through 4. analyze.py aggregates metrics and reports the first validation step at or above 90 percent. Mean test accuracy was 99.80 percent, sample standard deviation 0.27 percentage points. The original Step 1 streams were independently seeded but did not guarantee prompt disjointness, and online diagnostics used the test seed, so these results are exploratory. Sources: $repo/blob/main/STEP1_REPORT.md and the three cited Python modules."
    Add-Text $slide '5 independent seeds' 58 120 390 53 29 $true $teal | Out-Null
    Add-Text $slide '99.80% mean test accuracy' 58 188 800 59 33 $true $ink | Out-Null
    Add-Text $slide '0.27 percentage-point sample SD' 58 260 800 44 22 $false $muted | Out-Null
    Add-Text $slide 'The original Step 1 results are exploratory: prompt overlap was possible.' 58 365 835 80 22 $false $orange | Out-Null

    # 6. Base recursion
    $slide = New-Slide 'Base-task recursive training' 'project1/collapse.py: replacement_labels, run' "Generation zero uses clean labels. From generation one onward, replacement_labels samples the previous model's label distribution on a selected fraction of the fixed training prompts. run creates a newly initialized model and AdamW optimizer for each generation and selects a checkpoint using clean validation loss. The base pilot compares clean, 90 percent synthetic and 100 percent synthetic labels. Source: $repo/blob/main/project1/collapse.py"
    Add-Text $slide "labels, mask = replacement_labels(previous, train, fraction,`n                                  temperature, sampling_seed)`nset_seed(init_seed)`nmodel = InductionTransformer(config)" 57 134 846 161 20 $false $ink $codeFont | Out-Null
    Add-Text $slide 'Fixed prompts and dataset size; fresh weights and optimizer each generation.' 57 353 840 80 22 $false $teal | Out-Null

    # 7. Base pilot figure
    $slide = New-Slide 'Base pilot: no meaningful collapse' 'Figure: figures/step3-base/generation_comparison.png' "This is one seed, three generations and 1,024 clean test prompts. Clean and mixed training had identical labels and metrics. The fully synthetic chain had only one incorrect training label by generation two, with the same test accuracy as clean. The induction-attention score fell in all conditions, including clean, so that decline cannot be attributed to recursive feedback in this pilot. Source: $repo/blob/main/figures/step3-base/generation_comparison.png and $repo/blob/main/STEP3_REPORT.md"
    Add-Figure $slide (Join-Path $root 'figures\step3-base\generation_comparison.png') 150 100 660 390

    # 8. Extended task
    $slide = New-Slide 'Extended label, symbol, label task' 'project1/extended.py: oracle_split, sequence_logits, training_loss' "The model predicts the query label, then one of the two context symbols, then the context-specific label for that symbol. The true next-symbol conditional is uniform over the two context choices. sequence_logits applies a causal shift so the first prediction cannot see either later target. Clean rows use expected cross-entropy over both symbols; feedback rows use a sampled one-hot target. The total loss averages first-label, symbol and second-label terms. Source: $repo/blob/main/project1/extended.py"
    Add-Text $slide 'query label   next symbol   symbol label' 56 112 840 55 27 $true $ink $codeFont | Out-Null
    Add-Text $slide 'True next symbol: 50/50 over the two context symbols.' 56 189 835 55 22 $false $teal | Out-Null
    Add-Text $slide "first, symbol, second = sequence_logits(...)`nsymbol_loss = where(mask, sampled_loss, clean_loss).mean()`nloss = (CE(first) + symbol_loss + CE(second)) / 3" 56 297 845 137 19 $false $muted $codeFont | Out-Null

    # 9. Study protocol
    $slide = New-Slide 'Matched extended study' 'project1/extended_study.py; STEP2_REPORT.md' "extended_study.py runs five conditions for seeds zero, one and two. Every chain has three generations including generation zero, 4,096 training prompts, 512 prompts per original evaluation split and 3,000 updates per generation. Clean, 90 percent feedback and 100 percent feedback isolate retention. Soft oracle and sampled oracle resampling separate true-generator sampling from the one-hot target objective. All conditions share generation zero within a seed. Sources: $repo/blob/main/project1/extended_study.py and $repo/blob/main/STEP2_REPORT.md"
    Add-Line $slide 'clean' 'No replacement after generation 0' 111 270 18
    Add-Line $slide 'mixed' '90% model-generated triples' 174 270 18
    Add-Line $slide 'synthetic' '100% model-generated triples' 237 270 18
    Add-Line $slide 'resampled' 'Oracle continuation, soft symbol objective' 300 270 18
    Add-Line $slide 'resampled_sampled' 'Oracle continuation, sampled objective' 363 270 17
    Add-Text $slide '3 seeds, 3 generations, 45 selected checkpoints' 55 454 850 33 19 $true $ink | Out-Null

    # 10. Final held-out accuracy
    $slide = New-Slide 'Final test: label accuracy' 'Figure: figures/final-evaluation/comparison.png, upper panels' "The final evaluator generates 2,048 new test prompts per seed, excluding all original train, validation, test and diagnostic prompts. Checkpoints are frozen. At generation two, first-label accuracy means are 0.9982 clean, 0.9836 for 100 percent feedback, and 0.9904 for the matched sampled-oracle control. The second-label accuracy difference between feedback and matched control changes sign in one seed, so do not claim a stable accuracy gap. The source figure displays means and sample standard deviations over three seeds. Sources: $repo/blob/main/project1/final_evaluation.py and $repo/blob/main/figures/final-evaluation/comparison.png"
    Add-Figure $slide $finalPanels[0] 55 96 850 306
    Add-Text $slide 'Generation 2 first-label accuracy: 99.82% clean, 98.36% feedback, 99.04% matched oracle.' 55 423 850 61 20 $false $ink | Out-Null

    # 11. Final held-out symbol distribution
    $slide = New-Slide 'Final test: symbol distribution' 'Figure: figures/final-evaluation/comparison.png, lower panels' "Conditional total variation measures departure from the true 50/50 next-symbol distribution. At generation two it is 0.0028 clean, 0.1382 under 100 percent feedback and 0.1263 under sampled-oracle resampling. The matched oracle control therefore develops much of the skew without recursive model-generated targets. Global symbol coverage remains 50 of 50; this is conditional distribution drift, not loss of the symbol vocabulary. Sources: $repo/blob/main/figures/final-evaluation/comparison.png and $repo/blob/main/STEP3_REPORT.md"
    Add-Figure $slide $finalPanels[1] 55 96 850 306
    Add-Text $slide 'Most skew also appears with one-hot oracle targets; global symbol support remains intact.' 55 423 850 61 20 $false $ink | Out-Null

    # 12. Mechanistic attention and selected heads
    $slide = New-Slide 'Attention and selected-head ablation' 'Figure: figures/interpret-extended/mechanistic_comparison.png, upper panels' "interpret.py measures per-head attention, then ablates each head and the selected previous-token and induction pair. The mean maximum induction score at generation two is 0.475 clean and 0.444 under 100 percent feedback, but sampled-oracle is 0.432. The selected pair's mean accuracy drop is small: 0.029 clean and 0.055 feedback. This does not imply the second attention layer is unimportant; other heads may compensate. Sources: $repo/blob/main/project1/interpret.py and $repo/blob/main/figures/interpret-extended/mechanistic_comparison.png"
    Add-Figure $slide $mechanismPanels[0] 55 96 850 307
    Add-Text $slide 'The selected pair is not the whole circuit.' 55 427 850 50 22 $true $ink | Out-Null

    # 13. Layer ablation and patching
    $slide = New-Slide 'Layer ablation and activation patching' 'Figure: figures/interpret-extended/mechanistic_comparison.png, lower panels' "Ablating the entire second attention layer strongly reduces first-label accuracy: mean drop 0.820 clean and 0.656 under 100 percent feedback at generation two. Clean-to-corrupt patching of selected head outputs restores only part of the correct-label log-probability gap. Retaining only those two selected attention heads yields roughly 0.46 accuracy clean and 0.41 feedback, far below intact performance. These are causal tests but do not identify a minimal sufficient circuit. Sources: $repo/blob/main/project1/interpret.py and $repo/blob/main/STEP4_REPORT.md"
    Add-Figure $slide $mechanismPanels[1] 55 96 850 307
    Add-Text $slide 'The second attention layer matters; the selected pair is insufficient.' 55 427 850 50 21 $true $ink | Out-Null

    # 14. Symbol ablation
    $slide = New-Slide 'Symbol skew survives second-layer ablation' 'Figure: figures/interpret-extended/symbol_ablation.png' "The extended symbol output is evaluated before and after whole-second-layer attention ablation. At generation two, conditional TV in the 100 percent feedback model is about 0.139 intact and 0.131 ablated. The analogous sampled-oracle values are about 0.130 and 0.133. This intervention does not remove the skew, so the induction layer alone cannot explain it. Output weights, embeddings, first-layer processing and finite-sample target imbalance remain candidates. Source: $repo/blob/main/figures/interpret-extended/symbol_ablation.png and $repo/blob/main/STEP4_REPORT.md"
    Add-Figure $slide (Join-Path $root 'figures\interpret-extended\symbol_ablation.png') 70 118 820 320
    Add-Text $slide 'Second-layer induction heads do not uniquely cause the symbol skew.' 55 449 850 40 19 $false $ink | Out-Null

    # 15. Code map, baseline
    $slide = New-Slide 'Python files: baseline and model' 'Source: project1/ package' "data.py defines the symbolic task configuration, batch container and generator. model.py implements the two-layer causal transformer with inspectable and ablatable attention heads. diagnostics.py computes previous-token and induction attention scores and head-ablation changes. train.py runs baseline optimization and checkpoint selection. sweep.py launches independent seeds. analyze.py aggregates the baseline and detects the first 90 percent validation crossing. Sources: $repo/tree/main/project1"
    Add-Line $slide 'data.py' 'Task configuration and random prompt generator' 108 230 17
    Add-Line $slide 'model.py' 'Causal attention, transformer blocks, output head' 166 230 17
    Add-Line $slide 'diagnostics.py' 'Attention scores and single-head ablations' 224 230 17
    Add-Line $slide 'train.py' 'Baseline optimization, validation, checkpoints' 282 230 17
    Add-Line $slide 'sweep.py' 'Launch the five-seed baseline' 340 230 17
    Add-Line $slide 'analyze.py' 'Baseline summaries and phase-change plot' 398 230 17

    # 16. Code map, recursive experiments
    $slide = New-Slide 'Python files: recursive experiments' 'Source: project1/ package' "collapse.py implements fixed prompt partitioning, predecessor-label replacement and base generation chains. analyze_collapse.py audits paired base runs and plots results. extended.py defines the three-token oracle, autoregressive sampling, training loss, evaluation and target replacement. extended_study.py invokes the five-condition, three-seed protocol. analyze_extended.py audits targets and checkpoints and creates comparisons. final_evaluation.py builds a fresh excluded-prompt split and evaluates frozen checkpoints. interpret.py runs ablations, activation patching, representation statistics and mechanistic figures. Sources: $repo/tree/main/project1"
    Add-Line $slide 'collapse.py' 'Base label feedback and disjoint prompt splits' 92 285 16
    Add-Line $slide 'analyze_collapse.py' 'Base-run audit and plots' 147 285 16
    Add-Line $slide 'extended.py' 'Three-token targets, sampling, loss, evaluation' 202 285 16
    Add-Line $slide 'extended_study.py' 'Five matched conditions across three seeds' 257 285 16
    Add-Line $slide 'analyze_extended.py' 'Extended-run audit and comparison figures' 312 285 16
    Add-Line $slide 'final_evaluation.py' 'Fresh excluded-prompt test on frozen models' 367 285 16
    Add-Line $slide 'interpret.py' 'Head/layer ablation, patching, embeddings' 422 285 16

    # 17. Verification
    $slide = New-Slide 'Tests and reproducibility' 'tests/; README.txt; requirements.txt' "The repository contains nine test modules with 28 tests, last run successfully on 1 October 2026. They cover task generation, causal masking, feedback boundaries, split exclusion, protocol audits, analysis utilities and interpretation interventions. The repository tracks source and selected figures but ignores raw run artifacts and checkpoints. Rebuilding the mechanistic figures therefore requires retraining the saved protocol. A fresh-clone validation is still outstanding. Sources: $repo/tree/main/tests, $repo/blob/main/README.txt and $repo/blob/main/.gitignore"
    Add-Text $slide '28 tests passed in the last local run' 58 117 830 60 30 $true $teal | Out-Null
    Add-Text $slide "Core coverage: task invariants, causal masking, feedback targets,`nsplit isolation, protocol audits, and intervention behavior." 58 215 835 123 22 $false $ink | Out-Null
    Add-Text $slide 'Raw checkpoints are not on GitHub. Fresh-clone verification remains.' 58 377 830 75 21 $false $orange | Out-Null

    # 18. Conclusions and remaining work
    $slide = New-Slide 'Current conclusion and remaining work' 'STEP3_REPORT.md; STEP4_REPORT.md; PROJECT_CHECKLIST.md' "The strongest supported behavioral finding is a modest decline in clean-task performance under recursive feedback over two transitions. A matched sampled-oracle control reproduces much of the symbol-distribution skew, so the evidence does not support a feedback-only explanation. The selected head pair is not sufficient, while the second attention layer is necessary for high first-label performance. The study is a discrete-token adaptation rather than exact Omniglot replication. Remaining tasks are the course abstract, contribution and ethics statements, suitability confirmation, and fresh-clone packaging review. Sources: $repo/blob/main/STEP3_REPORT.md, $repo/blob/main/STEP4_REPORT.md and $repo/blob/main/PROJECT_CHECKLIST.md"
    Add-Text $slide 'Observed' 58 112 155 35 23 $true $teal | Out-Null
    Add-Text $slide 'Feedback lowers mean clean performance over two transitions.' 58 153 830 57 23 $false $ink | Out-Null
    Add-Text $slide 'Control' 58 245 155 35 23 $true $teal | Out-Null
    Add-Text $slide 'Sampled oracle targets explain much of the symbol skew.' 58 286 830 57 23 $false $ink | Out-Null
    Add-Text $slide 'Next' 58 376 155 35 23 $true $orange | Out-Null
    Add-Text $slide 'Confirm task fit, write the abstract, and test a fresh clone.' 58 417 830 61 23 $false $ink | Out-Null

    $deck.SaveAs($OutputPath, 24)
    for ($i = 1; $i -le $deck.Slides.Count; $i++) {
        $preview = Join-Path $build ('slide-{0:D2}.png' -f $i)
        $deck.Slides.Item($i).Export($preview, 'PNG', 1280, 720)
    }
    Write-Output "Saved $($deck.Slides.Count) slides to $OutputPath"
} finally {
    if ($deck -ne $null) { $deck.Close() }
    if ($ppt -ne $null) { $ppt.Quit() }
    if ($deck -ne $null) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($deck) }
    if ($ppt -ne $null) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
