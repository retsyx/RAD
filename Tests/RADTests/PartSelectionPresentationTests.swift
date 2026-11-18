import XCTest
@testable import RAD

final class PartSelectionPresentationTests: XCTestCase {
    func testPointOwnerRemainsPartialEvenWhenItIsTheOnlySolid() {
        let selection = PartSelectionPresentation(solidIndices: [4], selectedSolids: [],
                                                 pointSelectedSolids: [4], explicitlySelected: false)
        XCTAssertEqual(selection, .partial)
        XCTAssertTrue(selection.isSelected)
    }

    func testAllPointOwnersDoNotTurnAFolderIntoAWholePartSelection() {
        XCTAssertEqual(PartSelectionPresentation(solidIndices: [2, 4], selectedSolids: [],
                                                pointSelectedSolids: [2, 4], explicitlySelected: false), .partial)
    }

    func testWholePartSelectionTakesPrecedenceOverItsPoints() {
        XCTAssertEqual(PartSelectionPresentation(solidIndices: [4], selectedSolids: [4],
                                                pointSelectedSolids: [4], explicitlySelected: false), .full)
    }

    func testFolderIsPartialUntilAllItsPartsAreSelected() {
        XCTAssertEqual(PartSelectionPresentation(solidIndices: [2, 4], selectedSolids: [2],
                                                pointSelectedSolids: [], explicitlySelected: false), .partial)
        XCTAssertEqual(PartSelectionPresentation(solidIndices: [2, 4], selectedSolids: [2, 4],
                                                pointSelectedSolids: [], explicitlySelected: false), .full)
    }

    func testUnrelatedPointDoesNotSelectRowAndEmptyFoldersNeedExplicitSelection() {
        XCTAssertEqual(PartSelectionPresentation(solidIndices: [4], selectedSolids: [],
                                                pointSelectedSolids: [9], explicitlySelected: false), .none)
        XCTAssertEqual(PartSelectionPresentation(solidIndices: [], selectedSolids: [],
                                                pointSelectedSolids: [], explicitlySelected: false), .none)
        XCTAssertEqual(PartSelectionPresentation(solidIndices: [], selectedSolids: [],
                                                pointSelectedSolids: [], explicitlySelected: true), .full)
    }
}
