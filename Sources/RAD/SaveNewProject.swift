import Foundation

/// A separate design starts from one validated revision with a clean conversation.
enum SaveNewProject {
    static func snapshot(from original: RADProject, revisionID: UUID? = nil) throws -> RADProject {
        var revision: ProjectRevision
        if let revisionID {
            guard let requested = original.revisions.first(where: { $0.id == revisionID }) else {
                throw RADError.message("That design version is no longer available.")
            }
            revision = requested
        } else {
            guard let latest = original.revisions.last else {
                throw RADError.message("There is no design to save.")
            }
            revision = latest
        }
        revision.userMessageID = nil
        revision.responseMessageID = nil
        if revision.parts == nil, let preview = revision.preview {
            guard (1...4_096).contains(preview.solids) else {
                throw RADError.message("The saved design contains an invalid solid count.")
            }
            revision.parts = preview.parts ?? (0..<preview.solids).map {
                PartNode(id: "solid/\($0)", name: "Part \($0 + 1)", solidIndices: [$0])
            }
        }
        if revision.partLayout == nil { revision.partLayout = revision.parts.map { PartLayout(nodes: $0) } }
        revision.runtime = revision.runtime ?? revision.preview?.runtime
        revision.preview = nil
        return RADProject(messages: [], revisions: [revision], selectedRevisionID: revision.id,
                          projectID: UUID(), codexSettings: original.codexSettings,
                          projectName: original.projectName, showGrid: original.showGrid,
                          printVersionView: original.printVersionView, printLayoutView: original.printLayoutView,
                          projectionMode: original.projectionMode, displayMode: original.displayMode,
                          recentPartColors: original.recentPartColors,
                          partsPaneFraction: original.partsPaneFraction,
                          thumbnailData: revision.id == original.selectedRevisionID ? original.thumbnailData : nil)
    }
}
