#!/bin/bash

# MediSync Test Runner Script
# Provides easy execution of test suites

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo -e "${MAGENTA}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${MAGENTA}║       MediSync Test Suite Runner                 ║${NC}"
echo -e "${MAGENTA}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "$PROJECT_ROOT/medisync/__init__.py" ]; then
    echo -e "${RED}Error: Cannot find medisync package. Please run from project root.${NC}"
    exit 1
fi

# Function to run intensive test
run_intensive_test() {
    echo -e "${CYAN}Running Intensive Conversation Test...${NC}"
    echo ""
    cd "$PROJECT_ROOT"
    python -m medisync.tests.test_intensive_conversation
    TEST_EXIT_CODE=$?

    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Test completed successfully!${NC}"
    else
        echo ""
        echo -e "${RED}✗ Test failed with exit code: $TEST_EXIT_CODE${NC}"
    fi

    return $TEST_EXIT_CODE
}

# Function to check dependencies
check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"

    # Check Python version
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo -e "  Python version: ${GREEN}$PYTHON_VERSION${NC}"

    # Check for required packages
    python -c "import medisync" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "  MediSync package: ${GREEN}✓ Found${NC}"
    else
        echo -e "  MediSync package: ${RED}✗ Not found${NC}"
        return 1
    fi

    python -c "import qdrant_client" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "  Qdrant client: ${GREEN}✓ Found${NC}"
    else
        echo -e "  Qdrant client: ${RED}✗ Not found${NC}"
        return 1
    fi

    python -c "import rich" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "  Rich library: ${GREEN}✓ Found${NC}"
    else
        echo -e "  Rich library: ${RED}✗ Not found${NC}"
        return 1
    fi

    echo ""
    return 0
}

# Function to show menu
show_menu() {
    echo -e "${BLUE}Available Tests:${NC}"
    echo ""
    echo "  1) Intensive Conversation Test (Doctor-Patient, ~20 dialogues each)"
    echo "  2) Check Dependencies"
    echo "  3) Exit"
    echo ""
    echo -ne "${YELLOW}Select option [1-3]: ${NC}"
}

# Main execution
if [ "$1" == "--intensive" ] || [ "$1" == "-i" ]; then
    run_intensive_test
    exit $?
elif [ "$1" == "--check" ] || [ "$1" == "-c" ]; then
    check_dependencies
    exit $?
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  -i, --intensive    Run intensive conversation test"
    echo "  -c, --check        Check dependencies"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Run without arguments for interactive menu."
    exit 0
else
    # Interactive mode
    while true; do
        show_menu
        read choice

        case $choice in
            1)
                echo ""
                run_intensive_test
                echo ""
                echo -e "${CYAN}Press Enter to continue...${NC}"
                read
                echo ""
                ;;
            2)
                echo ""
                check_dependencies
                echo ""
                echo -e "${CYAN}Press Enter to continue...${NC}"
                read
                echo ""
                ;;
            3)
                echo ""
                echo -e "${GREEN}Goodbye!${NC}"
                exit 0
                ;;
            *)
                echo ""
                echo -e "${RED}Invalid option. Please select 1-3.${NC}"
                echo ""
                ;;
        esac
    done
fi
