import StoreKit
import SwiftUI

/// StoreKit is the source of truth for the lifetime entitlement. No local flag can grant it.
@MainActor
final class MembershipStore: ObservableObject {
    static let productID = "com.betabuddy.ios.lifetime"

    @Published private(set) var product: Product?
    @Published private(set) var isMember = false
    @Published private(set) var isBusy = false
    @Published var errorMessage: String?

    private var updatesTask: Task<Void, Never>?

    init() {
        updatesTask = Task { [weak self] in
            for await result in Transaction.updates {
                guard let self else { return }
                if case .verified(let transaction) = result {
                    await transaction.finish()
                }
                await self.refreshEntitlement()
            }
        }
        Task { await load() }
    }

    deinit { updatesTask?.cancel() }

    func load() async {
        await refreshEntitlement()
        do {
            product = try await Product.products(for: [Self.productID]).first
        } catch {
            product = nil
        }
    }

    func refreshEntitlement() async {
        var entitled = false
        for await result in Transaction.currentEntitlements {
            guard case .verified(let transaction) = result,
                  transaction.productID == Self.productID,
                  transaction.revocationDate == nil else { continue }
            entitled = true
            break
        }
        isMember = entitled
    }

    func purchase() async {
        guard let product, !isBusy else { return }
        isBusy = true
        defer { isBusy = false }
        do {
            switch try await product.purchase() {
            case .success(let result):
                guard case .verified(let transaction) = result else {
                    errorMessage = String(localized: "购买验证失败，请稍后重试。", bundle: AppLanguage.bundle)
                    return
                }
                await transaction.finish()
                await refreshEntitlement()
            case .pending, .userCancelled:
                break
            @unknown default:
                break
            }
        } catch {
            errorMessage = String(localized: "暂时无法完成购买，请稍后重试。", bundle: AppLanguage.bundle)
        }
    }

    func restore() async {
        guard !isBusy else { return }
        isBusy = true
        defer { isBusy = false }
        do {
            try await AppStore.sync()
            await refreshEntitlement()
            if !isMember {
                errorMessage = String(localized: "没有找到可恢复的终生会员购买。", bundle: AppLanguage.bundle)
            }
        } catch {
            errorMessage = String(localized: "暂时无法恢复购买，请稍后重试。", bundle: AppLanguage.bundle)
        }
    }
}
