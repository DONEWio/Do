#!/bin/bash
set -e

# Function to generate release notes block from two tags/commits.
generate_release_notes() {
    local from_tag="$1"
    local to_tag="$2"
    local header
    local repo_range
    local commits added fixed changed

    if git rev-parse "$to_tag" >/dev/null 2>&1; then
        header="## [${to_tag}] - $(date +'%Y-%m-%d')"
        repo_range="${from_tag}..${to_tag}"
    else
        header="## [HEAD] - $(date +'%Y-%m-%d')"
        repo_range="${from_tag}..HEAD"
    fi

    commits=$(git log ${repo_range} --pretty=format:"- %s")
    added=$(git log ${repo_range} --grep="^feat:" --pretty=format:"- %s")
    fixed=$(git log ${repo_range} --grep="^fix:" --pretty=format:"- %s")
    changed=$(git log ${repo_range} --grep="^refactor:\|^build:" --pretty=format:"- %s")

    echo "$header"
    echo ""
    echo "### Commits"
    echo "$commits"
    echo ""
    echo "### Added"
    echo "$added"
    echo ""
    echo "### Fixed"
    echo "$fixed"
    echo ""
    echo "### Changed"
    echo "$changed"
}

# Function to get tags in range (excludes the start tag, includes the end tag).
get_tags_in_range() {
    local start_tag="$1"
    local end_tag="$2"
    local found=0
    local range_tags=()
    for tag in $(git tag --sort=version:refname); do
        if [ "$tag" == "$start_tag" ]; then
            found=1
            continue
        fi
        if [ $found -eq 1 ]; then
            range_tags+=("$tag")
            if [ "$tag" == "$end_tag" ]; then
                break
            fi
        fi
    done
    echo "${range_tags[@]}"
}

CHANGELOG_FILE="CHANGELOG.md"
SUBTEXT="All notable changes to this project will be documented in this file\."

# Extract header from changelog file (up to and including the subtext line).
if grep -qE "^${SUBTEXT}$" "$CHANGELOG_FILE"; then
    HEADER_END=$(grep -nE "^${SUBTEXT}$" "$CHANGELOG_FILE" | cut -d: -f1 | head -n 1)
else
    HEADER_END=2
fi

HEADER_CONTENT=$(head -n "$HEADER_END" "$CHANGELOG_FILE")
BODY_CONTENT=$(tail -n +$((HEADER_END + 1)) "$CHANGELOG_FILE")

release_notes=""

if [ "$1" = "regen-all" ]; then
    # Regen-all mode: Get all tags using gh and regenerate release notes for all releases.
    all_tags=$(gh release list --limit 100 --json tagName --jq '.[].tagName' | sort -V)
    declare -a tags_array=()
    while IFS= read -r tag; do
        tags_array+=("$tag")
    done <<< "$all_tags"
    if [ ${#tags_array[@]} -eq 0 ]; then
        echo "Error: No releases found via gh."
        exit 1
    fi
    notes_list=()
    # Generate release notes for the initial release (from first commit to first tag)
    initial_commit=$(git rev-list --max-parents=0 HEAD)
    note=$(generate_release_notes "$initial_commit" "${tags_array[0]}")
    notes_list+=("$note")

    prev_tag="${tags_array[0]}"
    for tag in "${tags_array[@]:1}"; do
         note=$(generate_release_notes "$prev_tag" "$tag")
         notes_list+=("$note")
         prev_tag="$tag"
    done
    # Check for unreleased commits after the last tag
    if [ -n "$(git log ${prev_tag}..HEAD --pretty=format:'- %s')" ]; then
         note=$(generate_release_notes "$prev_tag" "HEAD")
         # Mark as unreleased
         note=$(echo "$note" | sed "s/## \[HEAD\]/## [Unreleased]/")
         notes_list+=("$note")
    fi
    # Combine notes in reverse order (latest on top)
    for (( idx=${#notes_list[@]}-1; idx>=0; idx-- )); do
         release_notes="${release_notes}\n${notes_list[idx]}\n"
    done

elif [ "$1" = "regen" ]; then
    shift
    for arg in "$@"; do
        case $arg in
            --from=*)
                from_tag="${arg#--from=}"
                ;;
            --to=*)
                to_tag="${arg#--to=}"
                ;;
            *)
                echo "Unknown argument: $arg"
                exit 1
                ;;
        esac
    done
    if [ -z "$from_tag" ]; then
        echo "Error: --from flag is required in regen mode."
        exit 1
    fi
    if [ -z "$to_tag" ]; then
        echo "Error: --to flag is required in regen mode."
        exit 1
    fi

    if git rev-parse "$from_tag" >/dev/null 2>&1; then
        if git rev-parse "$to_tag" >/dev/null 2>&1; then
            tags_in_range=($(get_tags_in_range "$from_tag" "$to_tag"))
            prev_tag="$from_tag"
            notes_list=()
            for tag in "${tags_in_range[@]}"; do
                note=$(generate_release_notes "$prev_tag" "$tag")
                notes_list+=("$note")
                prev_tag="$tag"
            done
            for (( idx=${#notes_list[@]}-1; idx>=0; idx-- )); do
                release_notes="${release_notes}\n${notes_list[idx]}\n"
            done
        else
            NEW_TAG="$to_tag"
            release_notes=$(generate_release_notes "$from_tag" "HEAD")
            release_notes=$(echo "$release_notes" | sed "s/## \[HEAD\]/## [${NEW_TAG}]/")
        fi
    else
        echo "Error: Starting tag '$from_tag' does not exist."
        exit 1
    fi

elif [[ "$1" == --release=* ]]; then
    NEW_TAG="${1#--release=}"
    last_release_line=$(echo "$BODY_CONTENT" | grep -m1 "^##")
    if [[ $last_release_line =~ \[([^]]+)\] ]]; then
        prev_tag="${BASH_REMATCH[1]}"
    else
        echo "Error: Could not determine previous release tag from changelog."
        exit 1
    fi
    release_notes=$(generate_release_notes "$prev_tag" "HEAD")
    release_notes=$(echo "$release_notes" | sed "s/## \[HEAD\]/## [${NEW_TAG}]/")

else
    echo "Usage: $0 regen-all
   OR:     $0 regen --from=x.x.x --to=y.y.y
   OR:     $0 --release=x.x.x"
    exit 1
fi

{
    echo "$HEADER_CONTENT"
    echo ""
    echo -e "$release_notes"
    echo ""
    echo "$BODY_CONTENT"
} > "$CHANGELOG_FILE"

echo "CHANGELOG.md has been updated with new release notes."