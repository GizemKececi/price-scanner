"""Manual price -> TRY and estimated tax-free price, without a camera or OCR."""
from __future__ import annotations

import html
from decimal import Decimal, ROUND_HALF_UP

import requests
import streamlit as st

from money_input import parse_price, parse_rate
from tax_rules import COUNTRIES, RULES_REVIEWED, estimate_refund


st.set_page_config(page_title="Price Converter", page_icon="💱", layout="centered")

st.markdown('''
<style>
 .block-container {max-width:630px; padding-top:1rem; padding-bottom:2rem;}
 header[data-testid="stHeader"] {height:0;min-height:0;background:transparent;}
 [data-testid="stVerticalBlock"] {gap:.65rem;}
 .pc-head {font-size:1.6rem;font-weight:750;letter-spacing:-.03em;margin-bottom:.05rem;}
 .pc-sub {color:var(--text-color);opacity:.7;font-size:.9rem;margin-bottom:1rem;}
 .pc-rate {border:1px solid rgba(135,135,135,.28);border-radius:12px;padding:.8rem 1rem;
           margin:.45rem 0 .6rem;}
 .pc-rate b {font-size:1.09rem;}
 .pc-muted {font-size:.79rem;opacity:.7;margin-top:.25rem;line-height:1.45;}
 .pc-grid {display:grid;grid-template-columns:1fr 1fr;gap:.65rem;margin-top:1rem;}
 .pc-card {border-radius:13px;border:1px solid rgba(135,135,135,.3);padding:.8rem .75rem;min-width:0;}
 .pc-label {font-size:.78rem;opacity:.76;min-height:2.3rem;line-height:1.3;}
 .pc-number {font-size:clamp(1.12rem,4.9vw,1.67rem);font-weight:770;white-space:normal;
             overflow-wrap:anywhere;line-height:1.3;margin-top:.4rem;letter-spacing:-.025em;}
 .pc-foot {font-size:.84rem;opacity:.81;margin-top:.65rem;line-height:1.6;}
 .pc-empty {padding:1.1rem 1rem;border:1px dashed rgba(135,135,135,.45);
             border-radius:13px;text-align:center;opacity:.75;margin-top:.8rem;}
 @media(max-width:370px){.pc-grid{gap:.4rem}.pc-card{padding:.6rem .45rem}}
</style>
''', unsafe_allow_html=True)


def tr_number(value: Decimal, digits: int = 2) -> str:
    unit = Decimal('1').scaleb(-digits)
    txt = f"{value.quantize(unit, rounding=ROUND_HALF_UP):,.{digits}f}"
    return txt.replace(',', '_').replace('.', ',').replace('_', '.')


def rate_str(value: Decimal) -> str:
    return format(value, 'f').replace('.', ',')


@st.cache_data(ttl=900, show_spinner=False)
def fx_reference(currency: str) -> tuple[Decimal, str]:
    response = requests.get(
        f'https://api.frankfurter.dev/v2/rate/{currency}/TRY', timeout=8,
    )
    response.raise_for_status()
    body = response.json()
    rate = Decimal(str(body['rate']))
    if rate <= 0:
        raise ValueError('Kur sıfır veya geçersiz')
    return rate, str(body['date'])


st.markdown('<div class="pc-head">💱 Price Converter</div>'
            '<div class="pc-sub">Fiyatı yaz, TL karşılığını ve tahmini tax free sonrasını gör.</div>',
            unsafe_allow_html=True)

def clear_price():
    st.session_state['typed_price'] = ''


country = st.selectbox(
    'Ülke', list(COUNTRIES), format_func=lambda c: COUNTRIES[c].label,
    key='country', on_change=clear_price,
)
rule = COUNTRIES[country]
currency = rule.currency

try:
    current_fx, fx_date = fx_reference(currency)
    fx_error = False
except (requests.RequestException, KeyError, TypeError, ValueError, ArithmeticError):
    current_fx, fx_date = None, None
    fx_error = True

price_key = 'typed_price'
st.text_input(
    f'Fiyatı yaz ({currency})', key=price_key,
    placeholder='Örn: 72.470' if currency in {'KRW', 'JPY'} else 'Örn: 1.299,90',
    help='KRW için 72.470 ve 72,470 aynı şekilde 72.470 olarak hesaplanır.',
)

st.button('↻ Fiyatı temizle', type='secondary', on_click=clear_price)

# Country defaults are preserved; the rate can be changed without touching app code.
with st.expander('⚙️ Tax free oranı ve kur ayarları'):
    st.caption(rule.summary)
    max_pct = float(rule.vat / (Decimal('100') + rule.vat) * Decimal('100'))
    can_refund = rule.method != 'unavailable'
    pct_key = f'custom_tax__{country}'
    manual_pct = st.checkbox('İade oranını kendim belirleyeceğim', key=pct_key, disabled=not can_refund)
    pct_value = None
    if manual_pct and can_refund:
        pct_value = st.number_input(
            'Etiket fiyatından iade edilecek oran (%)',
            min_value=0.0, max_value=max_pct,
            value=float(rule.estimated_net_pct) if rule.estimated_net_pct is not None else max_pct,
            step=0.5, format='%.2f', key=f'custom_tax_value__{country}',
        )
        st.caption('Vergi oranı değil, etiket fiyatına uygulanacak tahmini iade yüzdesidir.')
    custom_fx_key = f'custom_fx__{currency}'
    use_fx = st.checkbox('Kuru elle gireceğim', value=fx_error, key=custom_fx_key)
    manual_rate = None
    if use_fx:
        manual_rate = st.text_input(
            f'1 {currency} kaç TL?', key=f'custom_rate__{currency}',
            placeholder='Örn: 0,03697',
        )
    st.caption(rule.eligibility_note)
    st.caption(f'Tax free kuralları en son {RULES_REVIEWED} tarihinde kontrol edildi; mağaza ve işlem koşullarına göre değişebilir.')

rate = current_fx
if use_fx:
    try:
        rate = parse_rate(manual_rate or '')
    except ValueError as err:
        rate = None
        st.warning(str(err))

# Show the rate actually used for the calculation, including manual overrides.
st.markdown(
    '<div class="pc-rate">'
    f'<b>1 {html.escape(currency)} = '
    + (f'{html.escape(rate_str(rate))} TL' if rate is not None else 'Kur gerekli')
    + '</b><div class="pc-muted">'
    + ('Elle girilen kur' if use_fx and rate is not None
       else (f'Frankfurter · {html.escape(fx_date)} referans kuru · 15 dk önbellek'
             if fx_date and rate is not None
             else 'Canlı referans kura erişilemedi; ayarlardan kuru elle girebilirsin.'))
    + '</div></div>',
    unsafe_allow_html=True,
)

raw = st.session_state.get(price_key, '')
try:
    price = parse_price(raw, currency)
except ValueError as exc:
    price = None
    st.warning(str(exc))

if price is None:
    st.markdown('<div class="pc-empty">Hesaplamak için yukarıya ürünün fiyatını yaz.</div>',
                unsafe_allow_html=True)
elif rate is None:
    st.info('Önce bir kur gir; geçersiz veya bilinmeyen kurla hesaplama yapmıyorum.')
else:
    override = Decimal(str(pct_value)) if manual_pct and pct_value is not None else None
    result = estimate_refund(price, rule, override_pct=override)
    normal_tl = price * rate
    after_tl = (price - result.refund) * rate
    saving_tl = result.refund * rate
    if result.status == 'estimated':
        label = 'Tax free sonrası (tahmini)'
    elif result.status == 'maximum':
        label = 'Tax free sonrası (teorik)'
    elif result.status == 'ineligible':
        label = 'Tax free eşiğinin altında'
    else:
        label = 'Tax free uygulanmıyor'

    st.markdown(
        '<div class="pc-grid">'
        '<div class="pc-card"><div class="pc-label">Normal fiyat (TL)</div>'
        f'<div class="pc-number">{html.escape(tr_number(normal_tl))} TL</div></div>'
        f'<div class="pc-card"><div class="pc-label">{html.escape(label)}</div>'
        f'<div class="pc-number">{html.escape(tr_number(after_tl))} TL</div></div>'
        '</div>'
        '<div class="pc-foot">'
        f'Girilen fiyat: <b>{html.escape(tr_number(price, 0 if currency in {"KRW", "JPY"} else 2))} {html.escape(currency)}</b><br>'
        f'Tahmini iade: {html.escape(tr_number(result.refund, 0 if currency in {"KRW", "JPY"} else 2))} {html.escape(currency)}'
        f' ≈ {html.escape(tr_number(saving_tl))} TL'
        + ('<br>⚠️ Girilen ürünün fiyatı tek başına iade eşiğinin altında; aynı fişte başka ürünler varsa durum değişebilir.'
           if result.status == 'ineligible' else '')
        + ('<br>Bu tutar kesin iade değildir; operatör kesintisi veya mağaza koşulları sonucu değiştirebilir.'
           if result.status in {'estimated', 'maximum'} else '')
        + '</div>',
        unsafe_allow_html=True,
    )

st.caption('Kamera ve fotoğraf taraması yoktur. Kur gösterilen değerden hesaplanır; kart/banka kuru farklı olabilir.')
