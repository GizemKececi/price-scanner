"""Ülke bazlı turistik KDV iadesi hesapları.

NOT: Bunlar *internetten canlı gelen tax free oranları değildir*. Resmî
kaynaklarla doğrulanıp elle güncellenen, konservatif statik kurallardır.
Net iade, mağaza/operatör/fiş/ürün türüne bağlıdır; yerel para biriminde
hesaplanan tutarlar sadece tahmindir veya teorik üst sınırdır.
"""

from dataclasses import dataclass
from decimal import Decimal

RULES_REVIEWED = "2026-10-17"


@dataclass(frozen=True)
class TaxRule:
    label: str
    currency: str
    vat: Decimal
    min_receipt: Decimal | None
    method: str                 # estimate | ceiling | unavailable
    estimated_net_pct: Decimal | None
    summary: str
    eligibility_note: str
    sources: tuple[str, ...]


COUNTRIES: dict[str, TaxRule] = {
    "Güney Kore": TaxRule(
        label="🇰🇷 Güney Kore", currency="KRW", vat=Decimal("10"),
        min_receipt=Decimal("15000"), method="estimate",
        estimated_net_pct=Decimal("7"),
        summary="%7, operatör kesintileri için kullanılan örnek net iade varsayımıdır; resmî sabit oran değildir.",
        eligibility_note=(
            "15.000 KRW ve üzerindeki uygun alışverişlerde iade mümkündür. "
            "Mağaza Tax Free programına katılmalı. Anında iade için tek fiş "
            "1 milyon KRW'nin altında ve seyahat toplamı 5 milyon KRW'yi aşmamalıdır. "
            "Oran fiş ve hizmet sağlayıcısına göre değişebilir."
        ),
        sources=(
            "https://english.visitkorea.or.kr/svc/contents/contentsView.do?menuSn=489&vcontsId=248765",
            "https://www.law.go.kr/LSW/lsLawLinkInfo.do?chrClsCd=010202&lsJoLnkSeq=900329164",
        ),
    ),
    "Japonya": TaxRule(
        label="🇯🇵 Japonya", currency="JPY", vat=Decimal("10"),
        min_receipt=Decimal("5500"), method="ceiling", estimated_net_pct=None,
        summary="Standart %10 tüketim vergisinin fiyat içindeki teorik tutarı hesaplanır; operatör ücreti hariçtir.",
        eligibility_note=(
            "Standart vergili ürünlerde en az 5.000 JPY vergi hariç (5.500 JPY vergi dahil) "
            "aynı mağaza/gün alışveriş şartı vardır. 1 Kasım 2026'dan itibaren "
            "ülkeden çıkışta doğrulama sonrası iade sistemine geçilir. "
            "İndirimli %8 vergiye tabi ürünler farklıdır."
        ),
        sources=(
            "https://www.mlit.go.jp/kankocho/tax-free/page01_000001_00028.html",
            "https://www.japan.travel/en/us/blog/everything-you-need-to-know-japans-new-tax-free-shopping-system/",
        ),
    ),
    "Fransa": TaxRule(
        label="🇫🇷 Fransa", currency="EUR", vat=Decimal("20"),
        min_receipt=None, method="ceiling", estimated_net_pct=None,
        summary="Standart KDV için kesintisiz teorik iade tavanı gösterilir; net iade daha düşük olabilir.",
        eligibility_note="Uygunluk, fiş eşiği, ürün türü, mağaza ve iade şirketine bağlıdır; eşik bu sürümde doğrulanmıyor.",
        sources=(
            "https://www.douane.gouv.fr/en/fiche/tax-exemption-france-tourists-pablo",
            "https://europa.eu/youreurope/business/finance-and-tax/vat/vat-rules-rates/index_fr.htm",
        ),
    ),
    "Almanya": TaxRule(
        label="🇩🇪 Almanya", currency="EUR", vat=Decimal("19"),
        min_receipt=None, method="ceiling", estimated_net_pct=None,
        summary="Standart KDV için kesintisiz teorik iade tavanı gösterilir; net iade daha düşük olabilir.",
        eligibility_note="Uygunluk, fiş eşiği, ürün türü, mağaza ve iade şirketine bağlıdır; eşik bu sürümde doğrulanmıyor.",
        sources=(
            "https://europa.eu/youreurope/business/finance-and-tax/vat/vat-rules-rates/index_fr.htm",
        ),
    ),
    "İtalya": TaxRule(
        label="🇮🇹 İtalya", currency="EUR", vat=Decimal("22"),
        min_receipt=None, method="ceiling", estimated_net_pct=None,
        summary="Standart KDV için kesintisiz teorik iade tavanı gösterilir; net iade daha düşük olabilir.",
        eligibility_note="Uygunluk, fiş eşiği, ürün türü, mağaza ve iade şirketine bağlıdır; eşik bu sürümde doğrulanmıyor.",
        sources=(
            "https://europa.eu/youreurope/business/finance-and-tax/vat/vat-rules-rates/index_fr.htm",
        ),
    ),
    "Tayland": TaxRule(
        label="🇹🇭 Tayland", currency="THB", vat=Decimal("7"),
        min_receipt=Decimal("2000"), method="ceiling", estimated_net_pct=None,
        summary="%7 KDV'nin içindeki teorik üst sınırdır; fiilî iade resmî tutar tablosuna göre genellikle daha düşüktür.",
        eligibility_note="Aynı mağazada aynı gün en az 2.000 THB ve VAT Refund for Tourists programına katılan mağaza gerekir.",
        sources=(
            "https://vrtweb2.rd.go.th/38.html",
            "https://www.rd.go.th/fileadmin/user_upload/kortor/vat_refund/PDF/Vat-Refund-Sheetupdate.pdf",
        ),
    ),
    "Birleşik Krallık (İngiltere, İskoçya, Galler)": TaxRule(
        label="🇬🇧 Birleşik Krallık", currency="GBP", vat=Decimal("20"),
        min_receipt=None, method="unavailable", estimated_net_pct=None,
        summary="İngiltere, İskoçya ve Galler'de alışverişi valizde götüren turistler için genel VAT Retail Export Scheme bulunmuyor.",
        eligibility_note="Kuzey İrlanda farklı kurallara tabidir. Mağazadan yurt dışına doğrudan gönderim ayrı bir durum olabilir.",
        sources=(
            "https://www.gov.uk/guidance/claim-vat-back-on-tax-free-shopping-in-northern-ireland",
        ),
    ),
}


@dataclass(frozen=True)
class RefundResult:
    refund: Decimal
    pct_of_price: Decimal
    status: str  # estimated | maximum | ineligible | unavailable


def estimate_refund(amount: Decimal, rule: TaxRule, override_pct: Decimal | None = None) -> RefundResult:
    """Vergi dahil, tek ürün/fiş kabul edilen fiyattan yaklaşık tasarruf.

    override_pct etikete *göre* efektif iade yüzdesidir; VAT oranı değildir.
    """
    if amount <= 0:
        return RefundResult(Decimal("0"), Decimal("0"), "ineligible")
    if rule.method == "unavailable":
        return RefundResult(Decimal("0"), Decimal("0"), "unavailable")
    if rule.min_receipt is not None and amount < rule.min_receipt:
        return RefundResult(Decimal("0"), Decimal("0"), "ineligible")

    # Vergi dahil etiketteki gerçek KDV: vergi %10 ise 10 / 110.
    vat_ceiling = amount * rule.vat / (Decimal("100") + rule.vat)
    if override_pct is not None:
        refund = amount * max(Decimal("0"), override_pct) / Decimal("100")
        refund = min(refund, vat_ceiling)
        status = "estimated"
    elif rule.method == "estimate" and rule.estimated_net_pct is not None:
        refund = min(amount * rule.estimated_net_pct / Decimal("100"), vat_ceiling)
        status = "estimated"
    else:
        refund = vat_ceiling
        status = "maximum"
    pct = refund * Decimal("100") / amount
    return RefundResult(refund, pct, status)
