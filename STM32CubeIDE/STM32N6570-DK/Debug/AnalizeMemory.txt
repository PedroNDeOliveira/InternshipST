# Path to the .elf file
$elfFile = "x-cube-n6-ai-people-detection-tracking-dk.elf"

# Run arm-none-eabi-size and capture the output
$sizeOutput = & arm-none-eabi-size -A $elfFile

# Initialize total sizes
$internalRAMTotal = 0
$externalRAMTotal = 0

# Define sections allocated in internal RAM and external RAM
$internalSections = @(
    ".isr_vector",
    ".text",
    ".gnu.sgstubs",
    ".rodata",
    ".version",
    ".ARM.extab",
    ".ARM",
    ".preinit_array",
    ".init_array",
    ".fini_array",
    ".data",
    ".bss",
    "._user_heap_stack"
)

$externalSections = @(
    ".psram_section"
)

# Process each line of the size output
foreach ($line in $sizeOutput) {
    # Split the line by whitespace
    $parts = $line -split "\s+"
    
    if ($parts.Length -ge 2) {
        $section = $parts[0]
        $size = $parts[1] -as [int]

        if ($internalSections -contains $section) {
            $internalRAMTotal += $size
        } elseif ($externalSections -contains $section) {
            $externalRAMTotal += $size
        }
    }
}

# Convert to KB for readability
$internalRAMTotalKB = [math]::Round($internalRAMTotal / 1024, 2)
$externalRAMTotalKB = [math]::Round($externalRAMTotal / 1024, 2)

Write-Host "Total Internal RAM used: $internalRAMTotal bytes ($internalRAMTotalKB KB)"
Write-Host "Total External RAM used: $externalRAMTotal bytes ($externalRAMTotalKB KB)"
