import Foundation
@testable import RAD

/// The former single-file archive: archive JSON in a one-entry ZIP. The app
/// opens and saves directory packages only; tests and the one-time development
/// converter keep this codec.
extension ProjectStore {
    static func encode(_ project: RADProject) throws -> Data {
        try validate(project)
        let data = try ProjectZIP.encode(ProjectArchive.encode(project))
        guard data.count <= maximumFileBytes else { throw RADError.message("Invalid RAD design. File exceeds the 1,024 MB design limit.") }
        return data
    }

    static func decode(_ data: Data) throws -> RADProject {
        guard data.count <= maximumFileBytes else { throw RADError.message("Invalid RAD design. File exceeds the 1,024 MB design limit.") }
        let payload = ProjectZIP.isZIP(data) ? try ProjectZIP.decode(data) : data
        let project = try ProjectArchive.decode(payload)
        try validate(project)
        return project
    }
}
